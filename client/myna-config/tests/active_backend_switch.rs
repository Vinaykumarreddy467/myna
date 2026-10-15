use std::cell::RefCell;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::time::Duration;

use async_trait::async_trait;
use myna_config::active_backend::{
    ensure_backend_active, execute_switch, myna_restart_request, ActiveBackendController,
    PrepareSwitchError, Settled, SetupError, SetupStage, SnapdWait, SwitchNotice, SwitchOutcome,
    SwitchPlan,
};
use myna_config::backend_apply::ApplyPreview;
use myna_config::command::{CancellationToken, CommandRequest};
use myna_config::domain::{
    connection_snapshot, ActiveBackendState, BackendIdentity, BackendSnapshot, BackendSurface,
    BackendSurfaceError, CommandResult, ConnectionSnapshot, ContentConnections,
};
use myna_config::operation_gate::{OperationCoordinator, OperationKind};
use myna_config::ports::{
    BackendRepository, SystemConfigurator, SystemConfiguratorError, SystemConfiguratorFailure,
};
use myna_config::snap_changes::{parse_changes, ApplyProgress, ChangeInProgress};

/// What snapd lists: `slots` as `(snap, task)`, `myna:backend` on `connected`.
fn content_slots(slots: &[(&str, Option<&str>)], connected: &[&str]) -> ConnectionSnapshot {
    let slots: Vec<_> = slots
        .iter()
        .map(|(snap, task)| {
            let mut attrs = serde_json::json!({ "content": "inference-provider" });
            if let Some(task) = task {
                attrs["task"] = (*task).into();
            }
            let plugs = if connected.contains(snap) {
                serde_json::json!([{ "snap": "myna", "plug": "backend" }])
            } else {
                serde_json::json!([])
            };
            serde_json::json!({ "snap": snap, "slot": "provider", "attrs": attrs, "connections": plugs })
        })
        .collect();
    let connections: ContentConnections =
        serde_json::from_value(serde_json::json!({ "slots": slots })).unwrap();
    connection_snapshot(&connections)
}

fn connections(snaps: &[&str], connected: &[&str]) -> ConnectionSnapshot {
    let slots: Vec<_> = snaps
        .iter()
        .map(|snap| (*snap, Some("speech-to-text")))
        .collect();
    content_slots(&slots, connected)
}

fn argv(plan: &SwitchPlan) -> Vec<(&str, Vec<&str>)> {
    plan.operations()
        .iter()
        .map(|request| {
            (
                request.executable(),
                request.arguments().iter().map(String::as_str).collect(),
            )
        })
        .collect()
}

#[test]
fn plan_connects_from_zero_connections() {
    let plan = SwitchPlan::new(
        &connections(&["myna-parakeet"], &[]),
        BackendIdentity::new("myna-parakeet", "provider"),
    )
    .unwrap();
    assert_eq!(
        argv(&plan),
        [
            (
                "snap",
                vec!["connect", "myna:backend", "myna-parakeet:provider"]
            ),
            (
                "systemctl",
                vec!["--user", "restart", "snap.myna.myna.service"]
            )
        ]
    );
}

#[test]
fn plan_disconnects_a_provider_that_is_no_model() {
    // A machine onboarded before only speech-to-text counted had myna:backend
    // connected to gemma4, an LLM; the switch to a real model drops it.
    let snapshot = content_slots(&[("gemma4", None), ("myna-parakeet", None)], &["gemma4"]);
    assert_eq!(snapshot.active_state(), ActiveBackendState::Disconnected);
    let plan =
        SwitchPlan::new(&snapshot, BackendIdentity::new("myna-parakeet", "provider")).unwrap();
    assert_eq!(
        argv(&plan),
        [
            (
                "snap",
                vec!["disconnect", "myna:backend", "gemma4:provider"]
            ),
            (
                "snap",
                vec!["connect", "myna:backend", "myna-parakeet:provider"]
            ),
            (
                "systemctl",
                vec!["--user", "restart", "snap.myna.myna.service"]
            )
        ]
    );
}

#[test]
fn plan_switches_one_connection_disconnect_first() {
    let plan = SwitchPlan::new(
        &connections(&["myna-parakeet", "myna-whisper"], &["myna-parakeet"]),
        BackendIdentity::new("myna-whisper", "provider"),
    )
    .unwrap();
    assert_eq!(
        argv(&plan),
        [
            (
                "snap",
                vec!["disconnect", "myna:backend", "myna-parakeet:provider"]
            ),
            (
                "snap",
                vec!["connect", "myna:backend", "myna-whisper:provider"]
            ),
            (
                "systemctl",
                vec!["--user", "restart", "snap.myna.myna.service"]
            )
        ]
    );
}

#[test]
fn plan_disconnects_every_multiple_connection_before_connecting() {
    let plan = SwitchPlan::new(
        &connections(
            &["myna-parakeet", "myna-whisper", "other"],
            &["myna-parakeet", "myna-whisper"],
        ),
        BackendIdentity::new("other", "provider"),
    )
    .unwrap();
    assert_eq!(plan.operations().len(), 4);
    assert_eq!(plan.operations()[0].arguments()[0], "disconnect");
    assert_eq!(plan.operations()[1].arguments()[0], "disconnect");
    assert_eq!(plan.operations()[2].arguments()[0], "connect");
    assert_eq!(plan.operations()[3], myna_restart_request());
}

#[test]
fn same_backend_as_exactly_one_connection_is_noop() {
    let plan = SwitchPlan::new(
        &connections(&["myna-parakeet"], &["myna-parakeet"]),
        BackendIdentity::new("myna-parakeet", "provider"),
    )
    .unwrap();
    assert!(plan.operations().is_empty());
    assert!(plan.is_noop());
}

#[test]
fn selecting_one_of_multiple_connections_still_converges_to_one() {
    let plan = SwitchPlan::new(
        &connections(
            &["myna-parakeet", "myna-whisper"],
            &["myna-parakeet", "myna-whisper"],
        ),
        BackendIdentity::new("myna-parakeet", "provider"),
    )
    .unwrap();
    assert_eq!(
        argv(&plan),
        [
            (
                "snap",
                vec!["disconnect", "myna:backend", "myna-parakeet:provider"]
            ),
            (
                "snap",
                vec!["disconnect", "myna:backend", "myna-whisper:provider"]
            ),
            (
                "snap",
                vec!["connect", "myna:backend", "myna-parakeet:provider"]
            ),
            (
                "systemctl",
                vec!["--user", "restart", "snap.myna.myna.service"]
            ),
        ]
    );
}

#[test]
fn missing_selected_backend_is_rejected() {
    assert_eq!(
        SwitchPlan::new(
            &connections(&["myna-parakeet"], &[]),
            BackendIdentity::new("vanished", "provider"),
        )
        .unwrap_err(),
        PrepareSwitchError::BackendUnavailable(BackendIdentity::new("vanished", "provider"))
    );
}

type ChangesRead = Result<Vec<ChangeInProgress>, String>;

#[derive(Clone)]
struct FakeRepository {
    discoveries: Rc<RefCell<VecDeque<Result<ConnectionSnapshot, BackendSurfaceError>>>>,
    calls: Rc<RefCell<usize>>,
}

impl FakeRepository {
    fn new(
        discoveries: impl IntoIterator<Item = Result<ConnectionSnapshot, BackendSurfaceError>>,
    ) -> Self {
        Self {
            discoveries: Rc::new(RefCell::new(discoveries.into_iter().collect())),
            calls: Rc::new(RefCell::new(0)),
        }
    }

    fn calls(&self) -> usize {
        *self.calls.borrow()
    }
}

#[async_trait(?Send)]
impl BackendRepository for FakeRepository {
    async fn discover(
        &self,
        _cancellation: CancellationToken,
    ) -> Result<ConnectionSnapshot, BackendSurfaceError> {
        *self.calls.borrow_mut() += 1;
        self.discoveries.borrow_mut().pop_front().unwrap()
    }

    async fn read_snapshot(
        &self,
        backend: &BackendIdentity,
        _cancellation: CancellationToken,
    ) -> BackendSnapshot {
        BackendSnapshot::empty(backend.clone())
    }

    async fn refresh(
        &self,
        cancellation: CancellationToken,
    ) -> Result<ConnectionSnapshot, BackendSurfaceError> {
        self.discover(cancellation).await
    }
}

#[derive(Clone)]
struct FakeConfigurator {
    result: Rc<RefCell<Option<ConfiguratorResult>>>,
    calls: Rc<RefCell<Vec<Vec<CommandRequest>>>>,
    restarts: Rc<RefCell<usize>>,
    changes: Rc<RefCell<VecDeque<ChangesRead>>>,
    later: Rc<RefCell<VecDeque<ConfiguratorResult>>>,
}

type ConfiguratorResult = Result<Vec<CommandResult>, SystemConfiguratorFailure>;

impl FakeConfigurator {
    fn returning(result: ConfiguratorResult) -> Self {
        Self {
            result: Rc::new(RefCell::new(Some(result))),
            calls: Rc::new(RefCell::new(Vec::new())),
            restarts: Rc::new(RefCell::new(0)),
            changes: Rc::default(),
            later: Rc::default(),
        }
    }

    /// What the next switch executions return, after the first.
    fn then(self, result: ConfiguratorResult) -> Self {
        self.later.borrow_mut().push_back(result);
        self
    }

    /// What successive reads of snapd's changes report; once exhausted,
    /// nothing is in progress.
    fn with_changes(self, changes: impl IntoIterator<Item = ChangesRead>) -> Self {
        self.changes.borrow_mut().extend(changes);
        self
    }

    fn calls(&self) -> Vec<Vec<CommandRequest>> {
        self.calls.borrow().clone()
    }
}

#[async_trait(?Send)]
impl SystemConfigurator for FakeConfigurator {
    async fn user_daemons_enabled(&self, _cancellation: CancellationToken) -> Result<bool, String> {
        Ok(true)
    }

    async fn set_up(
        &self,
        _plan: &myna_config::ports::SetUpPlan,
        _cancellation: CancellationToken,
    ) -> Result<(), SystemConfiguratorError> {
        unreachable!("a backend switch never sets up")
    }

    async fn install_snap(
        &self,
        _snap: &str,
        _cancellation: CancellationToken,
    ) -> Result<Option<String>, SystemConfiguratorError> {
        unreachable!("a backend switch never installs a snap")
    }

    async fn snap_change(
        &self,
        _change_id: &str,
        _cancellation: CancellationToken,
    ) -> Result<ChangeInProgress, String> {
        unreachable!("a backend switch never follows an install")
    }

    async fn execute_backend_switch(
        &self,
        plan: &SwitchPlan,
        _cancellation: CancellationToken,
    ) -> Result<Vec<CommandResult>, SystemConfiguratorFailure> {
        self.calls.borrow_mut().push(plan.operations().to_vec());
        let first = self.result.borrow_mut().take();
        first
            .or_else(|| self.later.borrow_mut().pop_front())
            .unwrap()
    }

    async fn restart_myna(
        &self,
        _cancellation: CancellationToken,
    ) -> Result<(), SystemConfiguratorError> {
        *self.restarts.borrow_mut() += 1;
        match self.result.borrow_mut().take() {
            Some(Err(failure)) => Err(failure.into_parts().1),
            _ => Ok(()),
        }
    }

    async fn apply_backend_config(
        &self,
        _preview: &ApplyPreview,
        _cancellation: CancellationToken,
    ) -> Result<Vec<CommandResult>, SystemConfiguratorFailure> {
        unreachable!("a switch never applies backend settings")
    }

    async fn changes_in_progress(
        &self,
        _cancellation: CancellationToken,
    ) -> Result<Vec<ChangeInProgress>, String> {
        self.changes
            .borrow_mut()
            .pop_front()
            .unwrap_or(Ok(Vec::new()))
    }
}

fn success(plan: &SwitchPlan) -> Vec<CommandResult> {
    plan.operations()
        .iter()
        .map(|request| {
            CommandResult::new(
                request.executable(),
                request.arguments().to_vec(),
                Some(0),
                "",
                "",
            )
        })
        .collect()
}

fn error(message: &str) -> BackendSurfaceError {
    BackendSurfaceError::new(BackendSurface::Connections, message, "")
}

fn no_sleep(_: Duration) -> Pin<Box<dyn Future<Output = ()>>> {
    Box::pin(std::future::ready(()))
}

fn no_wait() -> SnapdWait<'static> {
    SnapdWait {
        interval: Duration::from_secs(2),
        timeout: Duration::from_secs(10),
        sleep: &no_sleep,
        cancellation: CancellationToken::new(),
    }
}

fn unheard(_: SetupStage) {}

fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
    gtk4::glib::MainContext::new().block_on(future)
}

#[test]
fn execution_rediscovers_before_and_after_and_reports_agreement() {
    let initial = connections(&["old", "new"], &["old"]);
    let final_state = connections(&["old", "new"], &["new"]);
    let plan = SwitchPlan::new(&initial, BackendIdentity::new("new", "provider")).unwrap();
    let repository = FakeRepository::new([Ok(initial), Ok(final_state.clone())]);
    let configurator = FakeConfigurator::returning(Ok(success(&plan)));

    let outcome = block_on(execute_switch(
        &plan,
        &configurator,
        &repository,
        CancellationToken::new(),
    ));

    assert!(matches!(
        outcome,
        SwitchOutcome::Applied { final_snapshot, .. }
            if final_snapshot == final_state
    ));
    assert_eq!(repository.calls(), 2);
    assert_eq!(configurator.calls(), [plan.operations()]);
}

#[test]
fn stale_discovery_and_disappearing_selection_are_blocked_without_privilege() {
    for current in [
        connections(&["old", "new", "external"], &["old", "external"]),
        connections(&["old"], &["old"]),
    ] {
        let original = connections(&["old", "new"], &["old"]);
        let plan = SwitchPlan::new(&original, BackendIdentity::new("new", "provider")).unwrap();
        let repository = FakeRepository::new([Ok(current.clone())]);
        let configurator = FakeConfigurator::returning(Ok(vec![]));
        let outcome = block_on(execute_switch(
            &plan,
            &configurator,
            &repository,
            CancellationToken::new(),
        ));
        assert!(matches!(
            outcome,
            SwitchOutcome::StaleDiscovery { final_snapshot } if final_snapshot == current
        ));
        assert!(configurator.calls().is_empty());
        assert_eq!(repository.calls(), 1);
    }
}

#[test]
fn cached_noop_is_rechecked_and_external_change_is_reported_without_privilege() {
    let cached = connections(&["old", "new"], &["new"]);
    let changed = connections(&["old", "new"], &["old"]);
    let plan = SwitchPlan::new(&cached, BackendIdentity::new("new", "provider")).unwrap();
    assert!(plan.is_noop());
    let repository = FakeRepository::new([Ok(changed.clone())]);
    let configurator = FakeConfigurator::returning(Ok(vec![]));

    let outcome = block_on(execute_switch(
        &plan,
        &configurator,
        &repository,
        CancellationToken::new(),
    ));

    assert!(matches!(
        outcome,
        SwitchOutcome::StaleDiscovery { final_snapshot } if final_snapshot == changed
    ));
    assert_eq!(repository.calls(), 1);
    assert!(configurator.calls().is_empty());
}

#[test]
fn verified_noop_returns_the_final_reread_state_without_authorization() {
    let cached = connections(&["old", "new"], &["new"]);
    let plan = SwitchPlan::new(&cached, BackendIdentity::new("new", "provider")).unwrap();
    let repository = FakeRepository::new([Ok(cached.clone())]);
    let configurator = FakeConfigurator::returning(Ok(vec![]));

    let outcome = block_on(execute_switch(
        &plan,
        &configurator,
        &repository,
        CancellationToken::new(),
    ));

    assert!(matches!(
        outcome,
        SwitchOutcome::Noop { final_snapshot } if final_snapshot == cached
    ));
    assert_eq!(repository.calls(), 1);
    assert!(configurator.calls().is_empty());
}

#[test]
fn every_operation_failure_and_auth_denial_still_rediscover_actual_state() {
    let initial = connections(&["old", "new"], &["old"]);
    let plan = SwitchPlan::new(&initial, BackendIdentity::new("new", "provider")).unwrap();
    let disconnected = connections(&["old", "new"], &[]);
    for (completed, error, final_state) in [
        (
            vec![],
            SystemConfiguratorError::execution("pkexec", vec![], Some(1), "disconnect", "failed"),
            initial.clone(),
        ),
        (
            success(&plan)[..1].to_vec(),
            SystemConfiguratorError::execution("pkexec", vec![], Some(1), "connect", "failed"),
            disconnected.clone(),
        ),
        (
            vec![],
            SystemConfiguratorError::authorization_denied("pkexec", vec![], Some(126), "denied"),
            initial.clone(),
        ),
    ] {
        // A disconnect that ran is followed by a read and an attempt to
        // put the old model back, before the final read.
        let restores = !completed.is_empty();
        let repository = FakeRepository::new(
            std::iter::once(Ok(initial.clone()))
                .chain(restores.then(|| Ok(disconnected.clone())))
                .chain([Ok(final_state.clone())]),
        );
        let configurator = FakeConfigurator::returning(Err(SystemConfiguratorFailure::new(
            completed.clone(),
            error.clone(),
        )))
        .then(Ok(Vec::new()));
        let outcome = block_on(execute_switch(
            &plan,
            &configurator,
            &repository,
            CancellationToken::new(),
        ));
        assert!(matches!(
            outcome,
            SwitchOutcome::Failed {
                completed: actual_completed,
                error: actual_error,
                final_snapshot: Some(actual),
                ..
            } if actual_completed == completed && actual_error == error && actual == final_state
        ));
        assert_eq!(repository.calls(), if restores { 3 } else { 2 });
    }
}

/// The user opened Settings again while snapd was still installing the model
/// and picked it: the disconnect would succeed and the connect be refused.
#[test]
fn a_switch_to_a_model_snapd_is_still_installing_changes_nothing() {
    let initial = connections(&["old", "myna-parakeet"], &["old"]);
    let plan =
        SwitchPlan::new(&initial, BackendIdentity::new("myna-parakeet", "provider")).unwrap();
    let repository = FakeRepository::new([Ok(initial.clone())]);
    let configurator = FakeConfigurator::returning(Ok(success(&plan))).with_changes([installing()]);

    let outcome = block_on(execute_switch(
        &plan,
        &configurator,
        &repository,
        CancellationToken::new(),
    ));

    assert_eq!(
        outcome,
        SwitchOutcome::Busy {
            change: "Install \"myna-parakeet\" snap from \"edge\" channel".to_owned(),
            final_snapshot: initial.clone(),
        }
    );
    assert_eq!(outcome.notice(), SwitchNotice::Failed);
    assert_eq!(outcome.final_snapshot(), Some(&initial));
    assert_eq!(configurator.calls(), Vec::<Vec<CommandRequest>>::new());
    assert_eq!(repository.calls(), 1);
}

#[test]
fn a_change_in_progress_on_another_snap_does_not_block_a_switch() {
    let initial = connections(&["old", "new"], &["old"]);
    let final_state = connections(&["old", "new"], &["new"]);
    let plan = SwitchPlan::new(&initial, BackendIdentity::new("new", "provider")).unwrap();
    let repository = FakeRepository::new([Ok(initial), Ok(final_state)]);
    let configurator = FakeConfigurator::returning(Ok(success(&plan)))
        .with_changes([change("Install \"unrelated\" snap")]);

    let outcome = block_on(execute_switch(
        &plan,
        &configurator,
        &repository,
        CancellationToken::new(),
    ));

    assert!(matches!(outcome, SwitchOutcome::Applied { .. }));
    assert_eq!(configurator.calls(), [plan.operations()]);
}

#[test]
fn an_unreadable_changes_list_does_not_block_a_switch() {
    let initial = connections(&["old", "new"], &["old"]);
    let final_state = connections(&["old", "new"], &["new"]);
    let plan = SwitchPlan::new(&initial, BackendIdentity::new("new", "provider")).unwrap();
    let repository = FakeRepository::new([Ok(initial), Ok(final_state)]);
    let configurator = FakeConfigurator::returning(Ok(success(&plan)))
        .with_changes([Err("snapd unreachable".to_owned())]);

    let outcome = block_on(execute_switch(
        &plan,
        &configurator,
        &repository,
        CancellationToken::new(),
    ));

    assert!(matches!(outcome, SwitchOutcome::Applied { .. }));
}

#[test]
fn a_failed_connect_reconnects_the_model_that_was_connected() {
    let initial = connections(&["old", "new"], &["old"]);
    let disconnected = connections(&["old", "new"], &[]);
    let restored = connections(&["old", "new"], &["old"]);
    let plan = SwitchPlan::new(&initial, BackendIdentity::new("new", "provider")).unwrap();
    let refused = SystemConfiguratorError::execution(
        "pkexec",
        vec![],
        Some(1),
        "connect",
        "changes in progress",
    );
    let repository =
        FakeRepository::new([Ok(initial), Ok(disconnected.clone()), Ok(restored.clone())]);
    let configurator = FakeConfigurator::returning(Err(SystemConfiguratorFailure::new(
        success(&plan)[..1].to_vec(),
        refused.clone(),
    )))
    .then(Ok(Vec::new()));

    let outcome = block_on(execute_switch(
        &plan,
        &configurator,
        &repository,
        CancellationToken::new(),
    ));

    let restore = SwitchPlan::new(&disconnected, BackendIdentity::new("old", "provider")).unwrap();
    assert_eq!(
        configurator.calls(),
        [plan.operations().to_vec(), restore.operations().to_vec()]
    );
    assert_eq!(
        argv(&restore),
        [
            ("snap", vec!["connect", "myna:backend", "old:provider"]),
            (
                "systemctl",
                vec!["--user", "restart", "snap.myna.myna.service"]
            ),
        ]
    );
    assert_eq!(restore.operations()[1], myna_restart_request());
    assert!(matches!(
        outcome,
        SwitchOutcome::Failed {
            error,
            final_snapshot: Some(actual),
            ..
        } if error == refused && actual == restored
    ));
    assert_eq!(repository.calls(), 3);
}

#[test]
fn a_failed_restore_still_reports_the_original_failure() {
    let initial = connections(&["old", "new"], &["old"]);
    let disconnected = connections(&["old", "new"], &[]);
    let plan = SwitchPlan::new(&initial, BackendIdentity::new("new", "provider")).unwrap();
    let refused = SystemConfiguratorError::execution("pkexec", vec![], Some(1), "connect", "busy");
    let repository = FakeRepository::new([
        Ok(initial),
        Ok(disconnected.clone()),
        Ok(disconnected.clone()),
    ]);
    let configurator = FakeConfigurator::returning(Err(SystemConfiguratorFailure::new(
        success(&plan)[..1].to_vec(),
        refused.clone(),
    )))
    .then(Err(SystemConfiguratorFailure::new(
        Vec::new(),
        SystemConfiguratorError::execution("pkexec", vec![], Some(1), "connect", "still busy"),
    )));

    let outcome = block_on(execute_switch(
        &plan,
        &configurator,
        &repository,
        CancellationToken::new(),
    ));

    assert_eq!(configurator.calls().len(), 2);
    assert!(matches!(
        outcome,
        SwitchOutcome::Failed {
            error,
            final_snapshot: Some(actual),
            ..
        } if error == refused && actual == disconnected
    ));
}

#[test]
fn no_model_is_reconnected_unless_a_disconnect_ran_and_one_model_was_connected() {
    let one = connections(&["old", "new"], &["old"]);
    let none = connections(&["old", "new"], &[]);
    let both = connections(&["old", "new"], &["old", "new"]);
    let new = BackendIdentity::new("new", "provider");
    let denied = || SystemConfiguratorError::authorization_denied("pkexec", vec![], Some(126), "");
    let failed = || SystemConfiguratorError::execution("pkexec", vec![], Some(1), "x", "failed");
    for (baseline, completed_ops, error) in [
        // A cancellation or denial is the user's choice, not a refusal.
        (one.clone(), 1, SystemConfiguratorError::Cancelled),
        (one.clone(), 1, denied()),
        // The disconnect itself failed: nothing to put back.
        (one.clone(), 0, failed()),
        // Nothing was connected before, so there is nothing to restore.
        (none.clone(), 1, failed()),
        // Several were connected: no single model to go back to.
        (both.clone(), 2, failed()),
    ] {
        let plan = SwitchPlan::new(&baseline, new.clone()).unwrap();
        let repository = FakeRepository::new([Ok(baseline.clone()), Ok(none.clone())]);
        let configurator = FakeConfigurator::returning(Err(SystemConfiguratorFailure::new(
            success(&plan)[..completed_ops].to_vec(),
            error,
        )));
        block_on(execute_switch(
            &plan,
            &configurator,
            &repository,
            CancellationToken::new(),
        ));
        assert_eq!(
            configurator.calls().len(),
            1,
            "{baseline:?} {completed_ops}"
        );
        assert_eq!(repository.calls(), 2);
    }
}

#[test]
fn setting_up_reports_a_model_still_installing_as_busy() {
    let snapshot = connections(&["myna-parakeet"], &[]);
    let repository = FakeRepository::new([Ok(snapshot.clone()), Ok(snapshot)]);
    let configurator =
        FakeConfigurator::returning(Ok(Vec::new())).with_changes([Ok(Vec::new()), installing()]);

    let result = block_on(ensure_backend_active(
        &repository,
        &configurator,
        None,
        &no_wait(),
        &unheard,
    ));

    assert_eq!(
        result.unwrap_err(),
        SetupError::Busy("Install \"myna-parakeet\" snap from \"edge\" channel".to_owned())
    );
    assert_eq!(configurator.calls(), Vec::<Vec<CommandRequest>>::new());
}

#[test]
fn cancellation_rediscovers_without_false_rollback() {
    let initial = connections(&["old", "new"], &["old"]);
    let plan = SwitchPlan::new(&initial, BackendIdentity::new("new", "provider")).unwrap();
    let final_state = connections(&["old", "new"], &[]);
    let repository = FakeRepository::new([Ok(initial.clone()), Ok(final_state.clone())]);
    let configurator = FakeConfigurator::returning(Err(SystemConfiguratorFailure::new(
        success(&plan)[..1].to_vec(),
        SystemConfiguratorError::Cancelled,
    )));
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let outcome = block_on(execute_switch(
        &plan,
        &configurator,
        &repository,
        cancellation,
    ));
    assert!(matches!(
        outcome,
        SwitchOutcome::Cancelled {
            final_snapshot: Some(actual),
            ..
        } if actual == final_state
    ));
    assert!(configurator.calls().is_empty());
}

#[test]
fn a_dismissed_authorization_prompt_is_a_cancel_that_rereads_the_connections() {
    let initial = connections(&["old", "new"], &["old"]);
    let plan = SwitchPlan::new(&initial, BackendIdentity::new("new", "provider")).unwrap();
    let repository = FakeRepository::new([Ok(initial.clone()), Ok(initial.clone())]);
    let configurator = FakeConfigurator::returning(Err(SystemConfiguratorFailure::new(
        Vec::new(),
        SystemConfiguratorError::Cancelled,
    )));
    let outcome = block_on(execute_switch(
        &plan,
        &configurator,
        &repository,
        CancellationToken::new(),
    ));
    assert!(matches!(
        outcome,
        SwitchOutcome::Cancelled {
            final_snapshot: Some(actual),
            discovery_error: None,
            ..
        } if actual == initial
    ));
    assert_eq!(repository.calls(), 2);
}

#[test]
fn partial_reconciliation_and_post_operation_disagreement_are_honest() {
    let initial = connections(&["old", "new", "external"], &["old", "external"]);
    let plan = SwitchPlan::new(&initial, BackendIdentity::new("new", "provider")).unwrap();
    let disagreement = connections(&["old", "new", "external"], &["external"]);
    let repository = FakeRepository::new([Ok(initial), Ok(disagreement.clone())]);
    let configurator = FakeConfigurator::returning(Ok(success(&plan)));

    let outcome = block_on(execute_switch(
        &plan,
        &configurator,
        &repository,
        CancellationToken::new(),
    ));
    assert!(matches!(
        outcome,
        SwitchOutcome::Disagreed { final_snapshot, .. } if final_snapshot == disagreement
    ));
}

#[test]
fn failed_final_rediscovery_is_exposed() {
    let initial = connections(&["old", "new"], &["old"]);
    let plan = SwitchPlan::new(&initial, BackendIdentity::new("new", "provider")).unwrap();
    let repository = FakeRepository::new([Ok(initial), Err(error("refresh failed"))]);
    let configurator = FakeConfigurator::returning(Ok(success(&plan)));
    assert!(matches!(
        block_on(execute_switch(
            &plan,
            &configurator,
            &repository,
            CancellationToken::new(),
        )),
        SwitchOutcome::FinalDiscoveryFailed { .. }
    ));
}

#[test]
fn controller_serializes_operations_ignores_stale_tokens_and_applies_final_discovery() {
    let initial = connections(&["old", "new"], &["old"]);
    let controller =
        ActiveBackendController::with_coordinator(initial.clone(), OperationCoordinator::new());
    let first = controller
        .begin(BackendIdentity::new("new", "provider"))
        .unwrap();
    assert_eq!(
        controller
            .begin(BackendIdentity::new("old", "provider"))
            .unwrap_err(),
        PrepareSwitchError::Busy
    );
    assert_eq!(
        controller
            .begin(BackendIdentity::new("new", "provider"))
            .unwrap_err(),
        PrepareSwitchError::Busy
    );
    let actual = connections(&["old", "new"], &["new"]);

    assert!(controller.complete(
        first.operation_token(),
        SwitchOutcome::Applied {
            completed: vec![],
            final_snapshot: initial.clone(),
        }
    ));
    let second = controller
        .begin(BackendIdentity::new("new", "provider"))
        .unwrap();
    assert!(!controller.complete(
        first.operation_token(),
        SwitchOutcome::Applied {
            completed: vec![],
            final_snapshot: initial,
        }
    ));
    assert!(controller.complete(
        second.operation_token(),
        SwitchOutcome::Applied {
            completed: vec![],
            final_snapshot: actual.clone(),
        }
    ));
    assert_eq!(
        controller.snapshot().active_state(),
        ActiveBackendState::Connected(BackendIdentity::new("new", "provider"))
    );
}

#[test]
fn controller_uses_the_shared_apply_switch_gate() {
    let gate = OperationCoordinator::new();
    let controller = ActiveBackendController::with_coordinator(
        connections(&["old", "new"], &["old"]),
        gate.clone(),
    );
    let apply = gate.begin(OperationKind::BackendApply).unwrap();
    assert_eq!(
        controller
            .begin(BackendIdentity::new("new", "provider"))
            .unwrap_err(),
        PrepareSwitchError::Busy
    );
    assert!(gate.complete(apply.token()));

    let switch = controller
        .begin(BackendIdentity::new("new", "provider"))
        .unwrap();
    assert!(gate.begin(OperationKind::BackendApply).is_err());
    assert_eq!(gate.active(), Some(OperationKind::BackendSwitch));
    assert!(gate.begin(OperationKind::BackendApply).is_err());
    assert!(controller.complete(
        switch.operation_token(),
        SwitchOutcome::Cancelled {
            completed: vec![],
            final_snapshot: Some(connections(&["old", "new"], &["old"])),
            discovery_error: None,
        }
    ));
    assert_eq!(gate.active(), None);
}

#[test]
fn selector_only_marks_a_backend_chosen_for_one_actual_connection() {
    let disconnected = ActiveBackendController::with_coordinator(
        connections(&["myna-parakeet", "myna-whisper"], &[]),
        OperationCoordinator::new(),
    );
    assert_eq!(disconnected.chosen(), None);

    let connected = ActiveBackendController::with_coordinator(
        connections(&["myna-parakeet", "myna-whisper"], &["myna-whisper"]),
        OperationCoordinator::new(),
    );
    assert_eq!(
        connected.chosen(),
        Some(BackendIdentity::new("myna-whisper", "provider"))
    );

    let multiple = ActiveBackendController::with_coordinator(
        connections(
            &["myna-parakeet", "myna-whisper"],
            &["myna-parakeet", "myna-whisper"],
        ),
        OperationCoordinator::new(),
    );
    assert_eq!(multiple.chosen(), None);
}

#[test]
fn a_pending_switch_marks_its_target_chosen_until_it_completes() {
    let controller = ActiveBackendController::with_coordinator(
        connections(&["myna-parakeet", "myna-whisper"], &["myna-parakeet"]),
        OperationCoordinator::new(),
    );
    let whisper = BackendIdentity::new("myna-whisper", "provider");
    assert_eq!(controller.switching_to(), None);
    let switch = controller.begin(whisper.clone()).unwrap();
    assert_eq!(controller.chosen(), Some(whisper.clone()));
    assert_eq!(controller.switching_to(), Some(whisper));

    assert!(controller.complete(
        switch.operation_token(),
        SwitchOutcome::Cancelled {
            completed: vec![],
            final_snapshot: Some(connections(
                &["myna-parakeet", "myna-whisper"],
                &["myna-parakeet"],
            )),
            discovery_error: None,
        }
    ));
    assert_eq!(
        controller.chosen(),
        Some(BackendIdentity::new("myna-parakeet", "provider"))
    );
    assert_eq!(controller.switching_to(), None);

    controller
        .begin(BackendIdentity::new("myna-whisper", "provider"))
        .unwrap();
    controller.abandon();
    assert_eq!(
        controller.chosen(),
        Some(BackendIdentity::new("myna-parakeet", "provider"))
    );
    assert_eq!(controller.switching_to(), None);
}

#[test]
fn only_a_switch_that_did_not_take_is_announced() {
    let snapshot = connections(&["myna-parakeet"], &["myna-parakeet"]);
    let notice = |outcome: SwitchOutcome| outcome.notice();
    let unread = || BackendSurfaceError::new(BackendSurface::Connections, "unreadable", "");
    assert_eq!(
        notice(SwitchOutcome::Applied {
            completed: Vec::new(),
            final_snapshot: snapshot.clone(),
        }),
        SwitchNotice::None
    );
    assert_eq!(
        notice(SwitchOutcome::Noop {
            final_snapshot: snapshot.clone(),
        }),
        SwitchNotice::None
    );
    assert_eq!(
        notice(SwitchOutcome::Cancelled {
            completed: Vec::new(),
            final_snapshot: None,
            discovery_error: Some(unread()),
        }),
        SwitchNotice::None
    );
    assert_eq!(
        notice(SwitchOutcome::Failed {
            completed: Vec::new(),
            error: SystemConfiguratorError::authorization_denied("snap", Vec::new(), None, ""),
            final_snapshot: Some(snapshot.clone()),
            discovery_error: None,
        }),
        SwitchNotice::Failed
    );
    assert_eq!(
        notice(SwitchOutcome::Disagreed {
            completed: Vec::new(),
            final_snapshot: snapshot.clone(),
        }),
        SwitchNotice::Failed
    );
    assert_eq!(
        notice(SwitchOutcome::StaleDiscovery {
            final_snapshot: snapshot,
        }),
        SwitchNotice::Failed
    );
    assert_eq!(
        notice(SwitchOutcome::FinalDiscoveryFailed {
            completed: Vec::new(),
            error: unread(),
        }),
        SwitchNotice::Unconfirmed
    );
}

/// A store install auto-connects a same-publisher backend, and the daemon may
/// have started before it existed: only the restart is left to do.
#[test]
fn an_auto_connected_backend_is_only_restarted() {
    let repository = FakeRepository::new([Ok(connections(&["myna-parakeet"], &["myna-parakeet"]))]);
    let configurator = FakeConfigurator::returning(Ok(vec![]));

    let settled = block_on(ensure_backend_active(
        &repository,
        &configurator,
        Some("myna-parakeet"),
        &no_wait(),
        &unheard,
    ))
    .unwrap();

    assert_eq!(settled, Settled::Restarted);
    assert!(configurator.calls().is_empty());
    assert_eq!(*configurator.restarts.borrow(), 1);
}

#[test]
fn a_failed_restart_is_reported() {
    let repository = FakeRepository::new([Ok(connections(&["myna-parakeet"], &["myna-parakeet"]))]);
    let configurator = FakeConfigurator::returning(Err(SystemConfiguratorFailure::new(
        Vec::new(),
        SystemConfiguratorError::execution("systemctl", vec![], Some(5), "", "unit not found"),
    )));

    let error = block_on(ensure_backend_active(
        &repository,
        &configurator,
        Some("myna-parakeet"),
        &no_wait(),
        &unheard,
    ))
    .unwrap_err()
    .to_string();

    assert!(error.contains("unit not found"), "{error}");
}

#[test]
fn an_unconnected_machine_switches_to_the_preferred_backend() {
    let initial = connections(&["myna-whisper", "myna-parakeet"], &[]);
    let connected = connections(&["myna-whisper", "myna-parakeet"], &["myna-parakeet"]);
    let repository = FakeRepository::new([Ok(initial.clone()), Ok(initial.clone()), Ok(connected)]);
    let plan =
        SwitchPlan::new(&initial, BackendIdentity::new("myna-parakeet", "provider")).unwrap();
    let configurator = FakeConfigurator::returning(Ok(success(&plan)));

    block_on(ensure_backend_active(
        &repository,
        &configurator,
        Some("myna-parakeet"),
        &no_wait(),
        &unheard,
    ))
    .unwrap();

    assert_eq!(configurator.calls(), [plan.operations()]);
    assert_eq!(*configurator.restarts.borrow(), 0);
}

#[test]
fn without_the_preferred_backend_the_first_discovered_one_is_used() {
    for preferred in [Some("myna-parakeet"), None] {
        let initial = connections(&["myna-whisper"], &[]);
        let connected = connections(&["myna-whisper"], &["myna-whisper"]);
        let repository =
            FakeRepository::new([Ok(initial.clone()), Ok(initial.clone()), Ok(connected)]);
        let plan =
            SwitchPlan::new(&initial, BackendIdentity::new("myna-whisper", "provider")).unwrap();
        let configurator = FakeConfigurator::returning(Ok(success(&plan)));

        block_on(ensure_backend_active(
            &repository,
            &configurator,
            preferred,
            &no_wait(),
            &unheard,
        ))
        .unwrap();

        assert_eq!(configurator.calls(), [plan.operations()], "{preferred:?}");
    }
}

#[test]
fn no_backend_and_failed_discovery_are_errors_without_privilege() {
    for discovery in [Ok(connections(&[], &[])), Err(error("snapd is down"))] {
        let repository = FakeRepository::new([discovery]);
        let configurator = FakeConfigurator::returning(Ok(vec![]));

        assert!(block_on(ensure_backend_active(
            &repository,
            &configurator,
            Some("myna-parakeet"),
            &no_wait(),
            &unheard,
        ))
        .is_err());
        assert!(configurator.calls().is_empty());
        assert_eq!(*configurator.restarts.borrow(), 0);
    }
}

#[test]
fn a_failed_or_contradicted_switch_is_reported() {
    let initial = connections(&["myna-parakeet"], &[]);
    let plan =
        SwitchPlan::new(&initial, BackendIdentity::new("myna-parakeet", "provider")).unwrap();
    let denied = FakeConfigurator::returning(Err(SystemConfiguratorFailure::new(
        Vec::new(),
        SystemConfiguratorError::snapd_authorization_denied(
            "POST /v2/interfaces",
            403,
            "cancelled",
        ),
    )));
    let repository = FakeRepository::new([
        Ok(initial.clone()),
        Ok(initial.clone()),
        Ok(initial.clone()),
    ]);
    // A refusal is reported as snapd's, naming its request; only a
    // dismissed prompt is not reported.
    assert!(matches!(
        block_on(ensure_backend_active(
            &repository,
            &denied,
            Some("myna-parakeet"),
            &no_wait(),
            &unheard,
        )),
        Err(SetupError::Step(_))
    ));

    // snapd reported success but the backend is still not connected.
    let contradicted = FakeConfigurator::returning(Ok(success(&plan)));
    let repository = FakeRepository::new([Ok(initial.clone()), Ok(initial.clone()), Ok(initial)]);
    let error = block_on(ensure_backend_active(
        &repository,
        &contradicted,
        Some("myna-parakeet"),
        &no_wait(),
        &unheard,
    ))
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("changed while it was being switched on"),
        "{error}"
    );
}

#[test]
fn a_switch_lost_to_discovery_or_cancellation_is_reported() {
    let initial = connections(&["myna-parakeet"], &[]);

    let repository = FakeRepository::new([Ok(initial.clone()), Err(error("snapd went away"))]);
    let configurator = FakeConfigurator::returning(Ok(vec![]));
    let lost = block_on(ensure_backend_active(
        &repository,
        &configurator,
        Some("myna-parakeet"),
        &no_wait(),
        &unheard,
    ))
    .unwrap_err()
    .to_string();
    assert!(lost.contains("snapd went away"), "{lost}");

    let repository = FakeRepository::new([Ok(initial.clone()), Ok(initial.clone()), Ok(initial)]);
    let configurator = FakeConfigurator::returning(Err(SystemConfiguratorFailure::new(
        Vec::new(),
        SystemConfiguratorError::Cancelled,
    )));
    let cancelled = block_on(ensure_backend_active(
        &repository,
        &configurator,
        Some("myna-parakeet"),
        &no_wait(),
        &unheard,
    ))
    .unwrap_err();
    assert_eq!(cancelled, SetupError::Cancelled);
}

/// The model's install change as snapd lists it while it downloads the snap,
/// after it has auto-connected `myna:backend`.
fn installing() -> ChangesRead {
    Ok(parse_changes(
        serde_json::from_str(include_str!("fixtures/snapd-changes-installing.json")).unwrap(),
    )
    .unwrap())
}

fn change(summary: &str) -> ChangesRead {
    Ok(parse_changes(serde_json::json!([{ "summary": summary }])).unwrap())
}

/// The wizard saw `myna:backend` connected while snapd was still installing
/// the model, restarted the daemon at once, and the daemon never found the
/// backend mount snapd applied seconds later.
#[test]
fn an_auto_connection_is_restarted_only_once_its_change_is_done() {
    let repository = FakeRepository::new([
        Ok(connections(&["myna-parakeet"], &["myna-parakeet"])),
        Ok(connections(&["myna-parakeet"], &["myna-parakeet"])),
    ]);
    let configurator =
        FakeConfigurator::returning(Ok(vec![])).with_changes([installing(), installing()]);
    let sleeps = RefCell::new(Vec::new());
    let sleep = |interval: Duration| -> Pin<Box<dyn Future<Output = ()>>> {
        sleeps
            .borrow_mut()
            .push((interval, *configurator.restarts.borrow()));
        Box::pin(std::future::ready(()))
    };
    let wait = SnapdWait {
        sleep: &sleep,
        ..no_wait()
    };

    block_on(ensure_backend_active(
        &repository,
        &configurator,
        Some("myna-parakeet"),
        &wait,
        &unheard,
    ))
    .unwrap();

    assert_eq!(
        *sleeps.borrow(),
        [(Duration::from_secs(2), 0), (Duration::from_secs(2), 0)]
    );
    assert_eq!(*configurator.restarts.borrow(), 1);
    assert!(configurator.changes.borrow().is_empty());
}

/// A spinner alone read as a hang while snapd fetched the model for minutes.
#[test]
fn every_stage_is_reported_as_it_starts() {
    let initial = connections(&["myna-parakeet"], &[]);
    let connected = connections(&["myna-parakeet"], &["myna-parakeet"]);
    let repository = FakeRepository::new([
        Ok(initial.clone()),
        Ok(initial.clone()),
        Ok(initial.clone()),
        Ok(connected),
    ]);
    let plan =
        SwitchPlan::new(&initial, BackendIdentity::new("myna-parakeet", "provider")).unwrap();
    let configurator = FakeConfigurator::returning(Ok(success(&plan))).with_changes([
        installing(),
        change("Connect myna:backend to myna-parakeet:provider"),
    ]);
    let stages = RefCell::new(Vec::new());

    block_on(ensure_backend_active(
        &repository,
        &configurator,
        Some("myna-parakeet"),
        &no_wait(),
        &|stage| stages.borrow_mut().push(stage),
    ))
    .unwrap();

    assert_eq!(
        *stages.borrow(),
        [
            SetupStage::Checking,
            SetupStage::Waiting(ApplyProgress::Download {
                name: "myna-parakeet".to_owned(),
                done: 52_428_800,
                total: 734_003_200,
            }),
            SetupStage::Waiting(ApplyProgress::Change {
                summary: "Connect myna:backend to myna-parakeet:provider".to_owned(),
            }),
            SetupStage::Checking,
            SetupStage::Connecting("myna-parakeet".to_owned()),
        ]
    );
}

#[test]
fn a_connected_backend_is_reported_as_restarting() {
    let repository = FakeRepository::new([Ok(connections(&["myna-parakeet"], &["myna-parakeet"]))]);
    let configurator = FakeConfigurator::returning(Ok(vec![]));
    let stages = RefCell::new(Vec::new());

    block_on(ensure_backend_active(
        &repository,
        &configurator,
        Some("myna-parakeet"),
        &no_wait(),
        &|stage| stages.borrow_mut().push(stage),
    ))
    .unwrap();

    assert_eq!(
        *stages.borrow(),
        [SetupStage::Checking, SetupStage::Restarting]
    );
}

/// Closing the wizard mid-wait must not connect or restart anything behind
/// it once snapd finishes.
#[test]
fn a_cancelled_wait_changes_nothing() {
    let repository = FakeRepository::new([Ok(connections(&["myna-parakeet"], &["myna-parakeet"]))]);
    let configurator =
        FakeConfigurator::returning(Ok(vec![])).with_changes([installing(), installing()]);
    let cancellation = CancellationToken::new();
    let sleep = |_: Duration| -> Pin<Box<dyn Future<Output = ()>>> {
        cancellation.cancel();
        Box::pin(std::future::ready(()))
    };
    let wait = SnapdWait {
        sleep: &sleep,
        cancellation: cancellation.clone(),
        ..no_wait()
    };

    let error = block_on(ensure_backend_active(
        &repository,
        &configurator,
        Some("myna-parakeet"),
        &wait,
        &unheard,
    ))
    .unwrap_err();

    assert_eq!(error, SetupError::Cancelled);
    assert_eq!(*configurator.restarts.borrow(), 0);
    assert!(configurator.calls().is_empty());
    assert_eq!(configurator.changes.borrow().len(), 1);
}

/// Before its install change auto-connects the backend, the wizard would
/// otherwise connect it itself and cost a polkit prompt.
#[test]
fn a_connection_its_change_makes_meanwhile_is_not_made_again() {
    let repository = FakeRepository::new([
        Ok(connections(&["myna-parakeet"], &[])),
        Ok(connections(&["myna-parakeet"], &["myna-parakeet"])),
    ]);
    let configurator = FakeConfigurator::returning(Ok(vec![])).with_changes([installing()]);

    block_on(ensure_backend_active(
        &repository,
        &configurator,
        Some("myna-parakeet"),
        &no_wait(),
        &unheard,
    ))
    .unwrap();

    assert!(configurator.calls().is_empty());
    assert_eq!(*configurator.restarts.borrow(), 1);
}

#[test]
fn changes_to_other_snaps_are_not_waited_for() {
    let repository = FakeRepository::new([Ok(connections(&["myna-parakeet"], &["myna-parakeet"]))]);
    let configurator = FakeConfigurator::returning(Ok(vec![]))
        .with_changes([change("Auto-refresh snap \"firefox\"")]);
    let sleep = |_: Duration| -> Pin<Box<dyn Future<Output = ()>>> {
        panic!("waited for a change to another snap")
    };
    let wait = SnapdWait {
        sleep: &sleep,
        ..no_wait()
    };

    block_on(ensure_backend_active(
        &repository,
        &configurator,
        Some("myna-parakeet"),
        &wait,
        &unheard,
    ))
    .unwrap();

    assert_eq!(*configurator.restarts.borrow(), 1);
}

#[test]
fn a_change_that_outlasts_the_wait_is_reported_without_a_restart() {
    let repository = FakeRepository::new([Ok(connections(&["myna-parakeet"], &["myna-parakeet"]))]);
    let configurator =
        FakeConfigurator::returning(Ok(vec![])).with_changes((0..10).map(|_| installing()));

    let error = block_on(ensure_backend_active(
        &repository,
        &configurator,
        Some("myna-parakeet"),
        &no_wait(),
        &unheard,
    ))
    .unwrap_err()
    .to_string();

    assert!(error.contains("Install \"myna-parakeet\" snap"), "{error}");
    assert_eq!(*configurator.restarts.borrow(), 0);
    // 10 s at 2 s: the first read, then one after each of five sleeps.
    assert_eq!(configurator.changes.borrow().len(), 4);
}

#[test]
fn unreadable_changes_are_reported_without_a_restart() {
    let repository = FakeRepository::new([Ok(connections(&["myna-parakeet"], &["myna-parakeet"]))]);
    let configurator =
        FakeConfigurator::returning(Ok(vec![])).with_changes([Err("snapd is down".to_owned())]);

    let failed = block_on(ensure_backend_active(
        &repository,
        &configurator,
        Some("myna-parakeet"),
        &no_wait(),
        &unheard,
    ))
    .unwrap_err()
    .to_string();

    assert!(failed.contains("snapd is down"), "{failed}");
    assert_eq!(*configurator.restarts.borrow(), 0);
}

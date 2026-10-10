import optuna
from playwright.sync_api import expect
from playwright.sync_api import Page
import pytest

from ...test_server import make_test_server


def make_test_storage() -> optuna.storages.InMemoryStorage:
    storage = optuna.storages.InMemoryStorage()
    sampler = optuna.samplers.RandomSampler(seed=0)

    study = optuna.create_study(study_name="single", storage=storage, sampler=sampler)

    def objective_single(trial: optuna.Trial) -> float:
        x1 = trial.suggest_float("x1", 0, 10)
        x2 = trial.suggest_float("x2", 0, 10)
        return (x1 - 2) ** 2 + (x2 - 5) ** 2

    study.optimize(objective_single, n_trials=50)

    return storage


@pytest.fixture
def storage() -> optuna.storages.InMemoryStorage:
    storage = make_test_storage()
    return storage


@pytest.fixture
def server_url(request: pytest.FixtureRequest, storage: optuna.storages.InMemoryStorage) -> str:
    return make_test_server(request, storage)


def test_history_xaxis_click(
    page: Page,
    storage: optuna.storages.InMemoryStorage,
    server_url: str,
) -> None:
    summaries = optuna.get_all_study_summaries(storage)
    study_id = summaries[0]._study_id
    study_name = summaries[0].study_name
    url = f"{server_url}/studies/{study_id}"

    page.goto(url)

    page.get_by_label("Datetime start").check()
    element = page.query_selector(".MuiTypography-body1")
    assert element is not None
    title = element.text_content()
    assert title is not None
    assert study_name in title

    page.get_by_label("Datetime complete").check()
    element = page.query_selector(".MuiTypography-body1")
    assert element is not None
    title = element.text_content()
    assert title is not None
    assert study_name in title

    page.get_by_label("Number").check()
    element = page.query_selector(".MuiTypography-body1")
    assert element is not None
    title = element.text_content()
    assert title is not None
    assert study_name in title


@pytest.mark.parametrize("log_scale,include_pruned", [(True, False), (True, True), (False, False)])
def test_history_filters_survive_navigation(
    page: Page,
    storage: optuna.storages.InMemoryStorage,
    server_url: str,
    log_scale: bool,
    include_pruned: bool,
) -> None:
    summary = optuna.get_all_study_summaries(storage)[0]
    study = optuna.load_study(study_name=summary.study_name, storage=storage)
    study.add_trial(
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.PRUNED, intermediate_values={0: 1.0}
        )
    )
    url = f"{server_url}/studies/{summary._study_id}"
    page.goto(url)

    log_scale_switch = page.get_by_label("Log y scale", exact=True)
    include_pruned_switch = page.get_by_label("Include PRUNED trials", exact=True)
    expect(log_scale_switch).not_to_be_checked()
    expect(include_pruned_switch).to_be_enabled()
    expect(include_pruned_switch).to_be_checked()
    log_scale_switch.set_checked(log_scale)
    include_pruned_switch.set_checked(include_pruned)

    for _ in range(2):
        page.get_by_title("Trials (List)", exact=True).get_by_role("link").click()
        expect(page).to_have_url(f"{url}/trials")
        expect(log_scale_switch).not_to_be_attached()
        page.get_by_title("History", exact=True).get_by_role("link").click()
        expect(page).to_have_url(url)
        expect(include_pruned_switch).to_be_enabled()
        expect(log_scale_switch).to_be_checked(checked=log_scale)
        expect(include_pruned_switch).to_be_checked(checked=include_pruned)

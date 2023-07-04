import nox


@nox.session(python=["3.9", "3.10", "3.11"], venv_backend="mamba")
def conda_tests(session: nox.Session) -> None:
    args = session.posargs or ["--cov"]
    # session.install("pytest", "pytest-cov")
    session.conda_install("pytest", "pytest-cov")
    session.conda_install("pytorch")
    session.conda_install("pydantic")
    session.install("-e", ".", "--no-deps")
    session.run("pytest", *args)

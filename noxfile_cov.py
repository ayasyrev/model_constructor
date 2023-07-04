import nox


@nox.session(python=["3.10"])
def cov_tests(session: nox.Session) -> None:
    args = session.posargs or ["--cov"]
    session.install(".", "pytest", "pytest-cov", "coverage[toml]")
    session.run("pytest", *args)


@nox.session(python="3.10")
def coverage(session: nox.Session) -> None:
    """Upload coverage data."""
    session.install("coverage[toml]", "codecov")
    session.run("coverage", "xml", "--fail-under=0")
    session.run("codecov", *session.posargs)

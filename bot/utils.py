import os

from exceptions import UnconfiguredEnvironmentError


def getenv_or_throw_exception(var_name):
    value = os.getenv(var_name)
    if not value:
        raise UnconfiguredEnvironmentError(var_name)
    return value


def is_running_in_docker():
    return os.path.exists('/.dockerenv')


def get_docker_secret(secret_name):
    with open(f'/run/secrets/{secret_name}', 'r') as f:
        return f.read().strip()
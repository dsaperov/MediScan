class UnconfiguredEnvironmentError(Exception):
    """This exception is raised if the requested environment variable is not set in the system."""

    def __init__(self, var_name):
        """
        Args:
            :param str var_name: name of the environment variable that could not be detected
        """
        msg = f'The environment variable {var_name} is not set in the system.'
        super().__init__(msg)
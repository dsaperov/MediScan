class UnconfiguredEnvironmentError(Exception):
    """This exception is raised if the requested environment variable is not set in the system."""

    def __init__(self, var_name):
        """
        Args:
            :param str var_name: name of the environment variable that could not be detected
        """
        msg = f'The environment variable {var_name} is not set in the system. Please execute the command ' \
              f'"cp .env.example .env" and set the values for the variables in the ".env" file, or ' \
              f'set these environment variables in the operating system in any other way.'
        super().__init__(msg)
from tcre.exec.v1 import cli
import os


def get_default_client():
    """Get CLI client that will not enforce setting of less important options (log level, balancing, seed, etc)"""
    return Client(require_options=True, exceptions=[
        'log_level', 'seed', 'vocab_limit', 'use_lower', 'save_keys',
        'log_iter_interval', 'log_epoch_interval', 'balance', 'batch_size',
        'simulation_strategy', 'swap_list'
    ])


class Client(object):

    def __init__(self, require_options=True, exceptions=None):
        """CLI wrapper client

        Example:

            from tcre.exec.v1 import cli_client
            client = cli_client.Client(require_options=True, exceptions=['log_level'])
            # Get executable command
            client.cmd(cli=dict(relation_class='test'), train=dict(dims=100))
            # Run and get return code for command
            client.run(cli=dict(relation_class='test'), train=dict(dims=100))

        """
        self.require_options = require_options
        self.exceptions = exceptions

    def cmd(self, executable='python', **kwargs):
        """Get runnable command"""
        cmd = [executable, cli.__file__]
        for fn_name, opts in kwargs.items():
            if fn_name != 'cli':
                cmd.append(fn_name)
            if self.require_options:
                missing = set(cli.PARAMS[fn_name]) - set(list(opts.keys())) - set(list(self.exceptions or []))
                if missing:
                    raise ValueError(f'Missing required options {missing} for command {fn_name}')
            for k, v in opts.items():
                cmd.append('--{}={}'.format(k.replace('_', '-'), v))
        return ' '.join(cmd)

    def run(self, raise_on_nonzero=True, **kwargs):
        """Run command for given options

        Returns:
            rc: Return code from os.system call
        """
        return self.execute(self.cmd(**kwargs), raise_on_nonzero=raise_on_nonzero)

    @classmethod
    def execute(cls, cmd, raise_on_nonzero=True):
        rc = os.system(cmd)
        if raise_on_nonzero and rc != 0:
            raise ValueError(f'Return code {rc} != 0 for command: {cmd}')
        return rc


import click

PARAMS = {}


class Client(object):
    
    def __init__(self, require_options=True, exceptions=None):
        self.require_options = require_options
        self.exceptions = exceptions
    
    def cmd(self, **kwargs):
        # cmd(dict(cli=dict(cli_opt=True), train=dict(train_opt='yes')))
        cmd = [__file__]
        for fn_name, opts in kwargs.items():
            if fn_name != 'cli':
                cmd.append(fn_name)
            if self.require_options:
                missing = set(PARAMS[fn_name]) - set(list(opts.keys())) - set(list(self.exceptions or []))
                if missing:
                    raise ValueError(f'Missing required options {missing} for command {fn_name}')
            for k, v in opts.items():
                cmd.append('--{}={}'.format(k.replace('_', '-'), v))
        return ' '.join(cmd)
            
        
class param(object):

    def __init__(self, *args, **kwargs):
        self.param = args[0].replace('--', '').replace('-', '_')
        self.click_fn = click.option(*args, **kwargs)

    def __call__(self, f):
        self.fn_name = f.__name__
        print(self.param, self.fn_name)
        if self.fn_name not in PARAMS:
            PARAMS[self.fn_name] = []
        PARAMS[self.fn_name].append(self.param)
        return self.click_fn(f)
    
@click.group(invoke_without_command=True)
@param('--cli-opt', default='cli', required=True)
@click.pass_context
def cli(ctx, cli_opt):
    print('In cli: cli_opt=', cli_opt)


@cli.command()
@param('--train-opt', default='train', required=True)
@click.pass_context
def train(ctx, train_opt):
    print('In train: train_opt=', train_opt)
    print('PARAMS = ', PARAMS)

if __name__ == '__main__':
    cli(obj={})
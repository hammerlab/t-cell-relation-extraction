
import click

@click.command()
@click.option('--use-test', default=True, type=bool)
def run(use_test):
    print(use_test)

if __name__ == '__main__':
    run()
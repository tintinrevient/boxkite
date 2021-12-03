import click
from boxkite_example.train import print_info

@click.command()
@click.option(
    "--start_date",
    type=click.DateTime(formats=['%Y-%m-%d']),
    required=True,
    help="Start date of this click CLI app"
)
def main(start_date):
    print_info(start_date)

if __name__ == "__main__":
    main()
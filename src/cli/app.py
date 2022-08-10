import click
from service import service


@click.group()
def cli():
    pass


@click.command()
def init_model():
    service.initialize_model()


@click.command()
@click.option('--file', prompt='Image to run prediction on', help='Image to predict')
def predict_img(file):
    print('file: ', file)
    service.predict_img(file)


cli.add_command(init_model)
cli.add_command(predict_img)

import connexion
from connexion.resolver import MethodViewResolver


app = connexion.FlaskApp(__name__, port=9090, specification_dir='openapi/')
app.add_api('openapi.yml', strict_validation=True, validate_responses=True,
            resolver=MethodViewResolver('views'))

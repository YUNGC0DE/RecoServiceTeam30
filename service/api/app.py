import logging

import uvicorn as uvicorn
from fastapi import FastAPI

from recommendations.estimator import discover_models
from recommendations.model_utils import load_model
from recommendations.resolvers import get_resolvers
from service.api.exception_handlers import add_exception_handlers
from service.api.middlewares import add_middlewares
from service.api.views import add_views
from service.log import setup_logging
from service.settings import ServiceConfig

log = logging.getLogger(__name__)

__all__ = ["create_app"]


def create_app(config: ServiceConfig) -> FastAPI:
    setup_logging(config)
    app = FastAPI(debug=False)
    app.state.k_recs = config.k_recs

    if config.resolution_strategy not in get_resolvers():
        raise ValueError(
            f"Wrong resolution strategy: {config.resolution_strategy}"
        )

    app.state.resolution_strategy = config.resolution_strategy
    log.info(f"Using resolution strategy: {config.resolution_strategy}")

    add_views(app)
    add_middlewares(app)
    add_exception_handlers(app)

    models = discover_models()
    for model in models.keys():
        load_model(model)

    return app


if __name__ == "__main__":
    from service.settings import get_config

    app = create_app(get_config())
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)

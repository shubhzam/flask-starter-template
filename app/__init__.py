from flask import Flask
from app.config import Config
from app.extensions import cors #db, migrate,
from app.errors import register_error_handlers
from app.routes.rag import rag_bp

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions
    # db.init_app(app)
    # migrate.init_app(app, db)
    cors.init_app(app)

    # Register blueprints & error handlers
    app.register_blueprint(rag_bp, url_prefix='/api/items')
    register_error_handlers(app)

    return app

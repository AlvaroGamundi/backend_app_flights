import os
from dotenv import load_dotenv
from sqlmodel import SQLModel, create_engine, Session

load_dotenv()  # carga las variables de entorno desde el archivo .env

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    raise ValueError("No se encontró DATABASE_URL en las variables de entorno")

engine = create_engine(DATABASE_URL, echo=True)

def crear_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session



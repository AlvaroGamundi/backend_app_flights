from sqlmodel import SQLModel, Field
from typing import Optional
from pydantic import BaseModel

class Usuario(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    nombre: str
    apellido: str
    email: str = Field(index=True, unique=True)
    contraseña: str

class UsuarioPublico(BaseModel):
    id: int
    nombre: str
    apellido: str
    email: str

    class Config:
        orm_mode = True
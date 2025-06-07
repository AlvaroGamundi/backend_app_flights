from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, validator
from sqlmodel import Session, select
import pandas as pd

from database import crear_db, get_session
from auth import registrar_usuario, autenticar_usuario, crear_token, obtener_usuario_desde_token

from src.limpiar_texto import LimpiadorTexto
from src.vectorizar import PreprocesadorBERT
from src.predecir import ModeloBERT
from src.respuesta import clase_respuesta
from src.user_table import Usuario

from typing import List
from src.user_table import UsuarioPublico

app = FastAPI()

# --- Configurar CORS ---
origins = [
    "http://localhost:3000",  # para pruebas locales frontend
    "https://frontend-app-flights.vercel.app",  # cambia por el dominio real de tu frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # o ["*"] para permitir todos (menos seguro)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ------------------------

# Crear la base de datos al iniciar (solo la primera vez)
@app.on_event("startup")
def on_startup():
    crear_db()

# Cargar modelo y componentes una sola vez
limpiador = LimpiadorTexto()
vectorizador = PreprocesadorBERT()
modelo = ModeloBERT("models/mejor_modelo_por_precision.pt", device='cpu')
respuesta_transformador = clase_respuesta()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# Schemas para registro y login
class RegistroUsuario(BaseModel):
    nombre: str
    apellido: str
    email: EmailStr
    contraseña: str

    @validator("contraseña")
    def contrasena_no_vacia(cls, v):
        if not v or not v.strip():
            raise ValueError("The password cannot be empty.")
        return v

class LoginUsuario(BaseModel):
    email: EmailStr
    contraseña: str

class TextoEntrada(BaseModel):
    texto: str

    @validator("texto")
    def texto_no_vacio(cls, v):
        if not str(v).strip():
            raise ValueError("The text cannot be empty.")
        return str(v)
    
@app.get("/usuarios", response_model=List[UsuarioPublico])
def listar_usuarios(session: Session = Depends(get_session)):
    usuarios = session.exec(select(Usuario)).all()
    return usuarios

@app.post("/registro")
def registro(usuario: RegistroUsuario, session: Session = Depends(get_session)):
    try:
        nuevo_usuario = registrar_usuario(session, usuario.nombre, usuario.apellido, usuario.email, usuario.contraseña)
        token = crear_token(nuevo_usuario)
        return {
            "mensaje": f"User '{nuevo_usuario.email}' created successfully.",
            "access_token": token,
            "token_type": "bearer"
        }
    except HTTPException as e:
        if e.status_code == 400:
            return {"error": e.detail}
        raise e

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), session: Session = Depends(get_session)):
    usuario = autenticar_usuario(session, form_data.username, form_data.password)
    if not usuario:
        raise HTTPException(status_code=400, detail="Incorrect email or password.")
    token = crear_token(usuario)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/predecir")
def predecir_sentimiento(
    entrada: TextoEntrada,
    usuario: Usuario = Depends(obtener_usuario_desde_token),
):
    texto_usuario = str(entrada.texto).strip()

    df = pd.DataFrame({'text': [texto_usuario]})
    df['text'] = df['text'].apply(limpiador.limpiar_texto)

    lista_texto = df['text'].tolist()
    input_ids, attention_masks = vectorizador.preprocesar(lista_texto)

    pred = modelo.predecir(input_ids, attention_masks)
    salida = respuesta_transformador(pred)

    return {"sentimiento": salida, "usuario": usuario.email}

# prueba semgrep

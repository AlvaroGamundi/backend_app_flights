import os
from passlib.context import CryptContext
from sqlmodel import Session, select
from fastapi import HTTPException, Depends
from src.user_table import Usuario
from database import get_session
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer

# Carga configuraciones desde variables de entorno
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("Falta definir la variable de entorno SECRET_KEY")

ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")  # Instancia fuera de la función

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def registrar_usuario(session: Session, nombre: str, apellido: str, email: str, contraseña: str) -> Usuario:
    usuario_existente = session.exec(select(Usuario).where(Usuario.email == email)).first()
    if usuario_existente:
        raise HTTPException(status_code=400, detail="Email ya registrado")

    usuario_nuevo = Usuario(
        nombre=nombre,
        apellido=apellido,
        email=email,
        contraseña=hash_password(contraseña)
    )

    session.add(usuario_nuevo)
    session.commit()
    session.refresh(usuario_nuevo)
    return usuario_nuevo

def autenticar_usuario(session: Session, email: str, contraseña: str) -> Usuario:
    usuario = session.exec(select(Usuario).where(Usuario.email == email)).first()
    if not usuario or not verify_password(contraseña, usuario.contraseña):
        return None
    return usuario

def crear_token(usuario: Usuario):
    datos = {"sub": usuario.email}
    expiracion = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    datos["exp"] = expiracion
    return jwt.encode(datos, SECRET_KEY, algorithm=ALGORITHM)

def obtener_usuario_desde_token(token: str = Depends(oauth2_scheme), session: Session = Depends(get_session)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Token inválido")
        
        usuario = session.exec(select(Usuario).where(Usuario.email == email)).first()
        if usuario is None:
            raise HTTPException(status_code=401, detail="Usuario no encontrado")
        
        return usuario
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido")




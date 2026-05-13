import os
import sys
from getpass import getpass
from app import create_app, db
from app.models import Usuario

app = create_app()

def main():
    print("=== Crear Nuevo Usuario ===")
    
    with app.app_context():
        # Asegurarnos de que las tablas existan (incluyendo usuarios)
        db.create_all()
        
        username = input("Nombre de usuario: ").strip()
        if not username:
            print("Error: El nombre de usuario no puede estar vacío.")
            return

        # Verificar si el usuario ya existe
        usuario_existente = Usuario.query.filter_by(username=username).first()
        if usuario_existente:
            print(f"Error: El usuario '{username}' ya existe.")
            return

        password = getpass("Contraseña: ")
        if not password:
            print("Error: La contraseña no puede estar vacía.")
            return

        password_confirm = getpass("Confirmar contraseña: ")
        if password != password_confirm:
            print("Error: Las contraseñas no coinciden.")
            return

        print("Selecciona el rol:")
        print("1. Administrador")
        print("2. Operador")
        rol_opcion = input("Opción (por defecto 1): ").strip()
        
        rol = "operador"
        if rol_opcion == "" or rol_opcion == "1":
            rol = "admin"

        nuevo_usuario = Usuario(username=username, rol=rol)
        nuevo_usuario.set_password(password)

        try:
            db.session.add(nuevo_usuario)
            db.session.commit()
            print(f"\n[ÉXITO] Usuario '{username}' creado exitosamente con rol de '{rol}'.")
        except Exception as e:
            db.session.rollback()
            print(f"\n[ERROR] Hubo un problema al crear el usuario: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperación cancelada.")
        sys.exit(0)

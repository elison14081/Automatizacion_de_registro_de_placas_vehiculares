from app import create_app, db
from app.models import Tarifa, Configuracion

app = create_app()

with app.app_context():
    # Crear todas las tablas
    db.create_all()
    
    # Verificar si ya existen tarifas
    if Tarifa.query.count() == 0:
        print("Insertando tarifas iniciales...")
        tarifas = [
            Tarifa(minutos_desde=0, minutos_hasta=60, precio=5.00, descripcion='Primera hora', activo=True),
            Tarifa(minutos_desde=61, minutos_hasta=120, precio=8.00, descripcion='Segunda hora', activo=True),
            Tarifa(minutos_desde=121, minutos_hasta=180, precio=10.00, descripcion='Tercera hora', activo=True),
            Tarifa(minutos_desde=181, minutos_hasta=None, precio=15.00, descripcion='Tarifa diaria (más de 3 horas)', activo=True),
        ]
        db.session.add_all(tarifas)
        db.session.commit()
        print("✓ Tarifas insertadas")
    
    # Verificar si ya existe configuración
    if Configuracion.query.count() == 0:
        print("Insertando configuración inicial...")
        configs = [
            Configuracion(clave='n8n_webhook_url', valor='https://TU_N8N_HOST/webhook/qr_whatsapp', descripcion='URL del webhook de n8n para WhatsApp'),
            Configuracion(clave='moneda', valor='S/', descripcion='Símbolo de moneda'),
            Configuracion(clave='nombre_estacionamiento', valor='Estacionamiento Central', descripcion='Nombre del negocio'),
            Configuracion(clave='tarifa_minima', valor='5.00', descripcion='Tarifa mínima de cobro'),
        ]
        db.session.add_all(configs)
        db.session.commit()
        print("✓ Configuración insertada")
    
    print("\n✅ Base de datos inicializada correctamente")
    print(f"Tarifas registradas: {Tarifa.query.count()}")
    print(f"Configuraciones registradas: {Configuracion.query.count()}")
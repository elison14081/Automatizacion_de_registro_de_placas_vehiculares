from app import create_app, db
from app.models import Movimiento, Tarifa, Configuracion

app = create_app()

@app.shell_context_processor
def make_shell_context():
    return {
        'db': db,
        'Movimiento': Movimiento,
        'Tarifa': Tarifa,
        'Configuracion': Configuracion
    }

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("✓ Tablas creadas correctamente")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
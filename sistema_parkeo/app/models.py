from . import db
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

class Movimiento(db.Model):
    __tablename__ = 'movimientos'
    
    id = db.Column(db.Integer, primary_key=True)
    placa = db.Column(db.String(10), nullable=False, index=True)
    token = db.Column(db.String(40), unique=True, nullable=False, index=True)
    hora_entrada = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    hora_salida = db.Column(db.DateTime, nullable=True)
    pagado = db.Column(db.Boolean, default=False, index=True)
    monto = db.Column(db.Float, default=0.0)
    qr_imagen = db.Column(db.Text, nullable=True)  # Base64 del QR
    numero_telefono = db.Column(db.String(15), nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'placa': self.placa,
            'token': self.token,
            'hora_entrada': self.hora_entrada.isoformat() if self.hora_entrada else None,
            'hora_salida': self.hora_salida.isoformat() if self.hora_salida else None,
            'pagado': self.pagado,
            'monto': self.monto,
            'numero_telefono': self.numero_telefono
        }

class Tarifa(db.Model):
    __tablename__ = 'tarifas'
    
    id = db.Column(db.Integer, primary_key=True)
    minutos_desde = db.Column(db.Integer, nullable=False)
    minutos_hasta = db.Column(db.Integer, nullable=True)
    precio = db.Column(db.Float, nullable=False)
    descripcion = db.Column(db.String(100), nullable=True)
    activo = db.Column(db.Boolean, default=True)
    
class Configuracion(db.Model):
    __tablename__ = 'configuracion'
    
    id = db.Column(db.Integer, primary_key=True)
    clave = db.Column(db.String(50), unique=True, nullable=False)
    valor = db.Column(db.String(255), nullable=False)
    descripcion = db.Column(db.String(200), nullable=True)

class Usuario(db.Model):
    __tablename__ = 'usuarios'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    rol = db.Column(db.String(20), default='operador')  # 'admin' u 'operador'
    activo = db.Column(db.Boolean, default=True)
    creado_en = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<Usuario {self.username} ({self.rol})>'
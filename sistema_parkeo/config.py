import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Configuración de Base de Datos (SQLite para desarrollo local)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or f"sqlite:///{os.path.join(BASE_DIR, 'app.db')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Configuración de uploads
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'static', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

    # Configuración de WhatsApp Business API
    WHATSAPP_TOKEN = os.environ.get('WHATSAPP_TOKEN')
    WHATSAPP_PHONE_ID = os.environ.get('WHATSAPP_PHONE_ID')
    
    # Configuración de Twilio WhatsApp
    TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
    TWILIO_WHATSAPP_FROM = os.environ.get('TWILIO_WHATSAPP_FROM', 'whatsapp:+14155238886')
    
    # Modelo YOLO
    YOLO_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelo_placa.pt')

    # Números permitidos en modo desarrollo (agrégalos en Meta for Developers)
    NUMEROS_PERMITIDOS_DEV = [
        '51922394409',  # Tu número verificado
        # Agrega más números aquí después de verificarlos en Meta
    ]
    
    # Modo de desarrollo para WhatsApp
    WHATSAPP_DEV_MODE = os.environ.get('WHATSAPP_DEV_MODE', 'true').lower() == 'true'
import qrcode
import base64
import json
import secrets
from io import BytesIO
from datetime import datetime
import io

class QRService:
    
    @staticmethod
    def generar_qr_movimiento(placa, token, hora_entrada, numero_telefono=None):
        """Genera un QR con los datos del movimiento"""
        
        # Datos a codificar en el QR
        data = {
            "placa": placa,
            "token": token,
            "hora_entrada": hora_entrada.timestamp(),
            "telefono": numero_telefono
        }
        
        data_str = json.dumps(data)
        
        # Crear QR
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=2,
        )
        qr.add_data(data_str)
        qr.make(fit=True)
        
        # Generar imagen
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convertir a base64
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        qr_img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return qr_img_b64
    
    @staticmethod
    def decodificar_qr(qr_data_str):
        """Decodifica los datos de un QR escaneado"""
        try:
            data = json.loads(qr_data_str)
            return {
                'placa': data.get('placa'),
                'token': data.get('token'),
                'hora_entrada': datetime.fromtimestamp(data.get('hora_entrada')),
                'telefono': data.get('telefono')
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
    
    @staticmethod
    def generar_token():
        """Genera un token único para el movimiento"""
        return secrets.token_hex(16)
    
    def generar_qr_movimiento(self, placa, token, hora_entrada, numero_telefono=None):
        """
        Genera un código QR con la URL de consulta del movimiento
        """
        # URL que el usuario verá al escanear
        url_consulta = f"http://localhost:5000/mi-ticket/{token}"
        
        # Crear QR
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        
        qr.add_data(url_consulta)
        qr.make(fit=True)
        
        # Generar imagen
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convertir a base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return img_base64
    
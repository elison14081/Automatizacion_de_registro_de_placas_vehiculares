import qrcode
import base64
import json
import secrets
from datetime import datetime
import io
import qrcode.image.svg

class QRService:
    @staticmethod
    def _crear_qr_data_uri(data):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)

        try:
            img = qr.make_image(fill_color="black", back_color="white")
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f'data:image/png;base64,{encoded}'
        except ImportError:
            factory = qrcode.image.svg.SvgPathImage
            img = qr.make_image(image_factory=factory)
            buffer = io.BytesIO()
            img.save(buffer)
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f'data:image/svg+xml;base64,{encoded}'
    
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
        
        return QRService._crear_qr_data_uri(data_str)
    
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
        
        return self._crear_qr_data_uri(url_consulta)
    

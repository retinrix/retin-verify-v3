"""WebSocket routes for real-time verification."""

import json
import base64
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import cv2
import logging

from api.services.pipeline_manager import PipelineConfig

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """WebSocket connection manager."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_progress(self, websocket: WebSocket, data: dict):
        await websocket.send_json(data)


manager = ConnectionManager()


@router.websocket("/verify")
async def websocket_verify(websocket: WebSocket):
    """
    WebSocket endpoint for real-time verification.
    
    Receives:
    - {"type": "start", "image": "base64...", "options": {...}}
    
    Sends:
    - {"type": "progress", "stage": "...", "progress": 0.5}
    - {"type": "complete", "result": {...}}
    - {"type": "error", "message": "..."}
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data.get('type') == 'start':
                # Decode image
                try:
                    image_data = data['image']
                    if ',' in image_data:
                        image_data = image_data.split(',')[1]
                    
                    image_bytes = base64.b64decode(image_data)
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image is None:
                        await manager.send_progress(websocket, {
                            'type': 'error',
                            'message': 'Could not decode image'
                        })
                        continue
                    
                except Exception as e:
                    await manager.send_progress(websocket, {
                        'type': 'error',
                        'message': f'Image decode error: {str(e)}'
                    })
                    continue
                
                # Get options
                options = data.get('options', {})
                config = PipelineConfig(
                    enable_document_detection=options.get('detect_document', True),
                    enable_mrz_extraction=options.get('extract_mrz', True),
                    enable_face_extraction=options.get('extract_face', True),
                    enable_security_analysis=options.get('security_check', True),
                    enable_tampering_detection=options.get('tampering_check', True)
                )
                
                # Progress callback
                async def progress_callback(update: dict):
                    await manager.send_progress(websocket, update)
                
                # Get pipeline
                pipeline = websocket.app.state.pipeline
                
                # Run verification
                try:
                    result = await pipeline.run(
                        image,
                        config,
                        progress_callback=progress_callback
                    )
                    
                    # Send final result
                    await manager.send_progress(websocket, {
                        'type': 'complete',
                        'result': result
                    })
                    
                except Exception as e:
                    logger.error(f"Verification error: {e}")
                    await manager.send_progress(websocket, {
                        'type': 'error',
                        'message': str(e)
                    })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

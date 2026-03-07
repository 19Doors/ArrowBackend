# server.py
import os
from livekit import api
from flask import Flask, jsonify, request
from flask_cors import CORS  # Add this import

app = Flask(__name__)
CORS(app)

@app.route('/getToken', methods=['POST'])
def getToken():
    body = request.get_json()
    
    token = api.AccessToken(os.getenv('LIVEKIT_API_KEY'), os.getenv('LIVEKIT_API_SECRET'))
    
    # If this room doesn't exist, it'll be automatically created when
    # the first participant joins
    room_name = body.get('room_name') or 'quickstart-room'
    token = token.with_grants(api.VideoGrants(room_join=True, room=room_name))
    
    if body.get('room_config'):
        token = token.with_room_config(body['room_config'])
    
    # Participant related fields. 
    # `participantIdentity` will be available as LocalParticipant.identity
    # within the livekit-client SDK
    token = token.with_identity(body.get('participant_identity') or 'quickstart-identity')
    token = token.with_name(body.get('participant_name') or 'quickstart-username')
    if body.get('participant_metadata'):
        token = token.with_metadata(body['participant_metadata'])
    if body.get('participant_attributes'):
        token = token.with_attributes(body['participant_attributes'])
    
    return jsonify({
        'server_url': os.getenv('LIVEKIT_URL'),
        'participant_token': token.to_jwt()
    })

if __name__ == '__main__':
    app.run(port=3002)

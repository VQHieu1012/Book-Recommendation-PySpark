# Download the helper library from https://www.twilio.com/docs/python/install

import os
from twilio.rest import Client

# Set environment variables for your credentials
# Read more at http://twil.io/secure

from keys import ACCOUNT_SID, AUTH_TOKEN, TWILIO_NUMBER, MY_PHONE_NUMBER
from twilio.twiml.voice_response import VoiceResponse

def notify():


  account_sid = ACCOUNT_SID
  auth_token = AUTH_TOKEN
  client = Client(account_sid, auth_token)
  

  call = client.calls.create(
    url="http://demo.twilio.com/docs/classic.mp3",
    to= MY_PHONE_NUMBER,
    from_=TWILIO_NUMBER
  )

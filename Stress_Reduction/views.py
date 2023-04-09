from django.shortcuts import render
from django.http import HttpResponse
from tensorflow.keras.models import load_model
import numpy as np

import openai

# Set your API key
openai.api_key = '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'

def predict_stress(request):
    if request.method == 'POST':
        # Get the input values from the form
        respiration_rate = float(request.POST['respiration_rate'])
        limb_movement = float(request.POST['limb_movement'])
        blood_oxygen = float(request.POST['blood_oxygen'])
        sleeping_hours = float(request.POST['sleeping_hours'])
        heart_rate = float(request.POST['heart_rate'])


        prompt = str(request.POST['user_location'])

        # Call OpenAI's ChatGPT API
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": f"I'm already in the {prompt} and My respiration rate is {respiration_rate}, limb movement is {limb_movement}, blood oxygen is {blood_oxygen}, sleeping hours is {sleeping_hours}, and heart rate is {heart_rate}. Give me 3 remedies based on where I'm right now in bullets. Don't write 'Here are some suggestions: As an AI language model, I cannot determine the exact remedy you need. However,' in the response."},
            ]
        )

        # Extract the assistant's reply from the API response
        assistant_reply = response['choices'][0]['message']['content']


        # Load the trained model
        model = load_model(r'C:\Users\njain12\Desktop\FullyHack2023_Project\Stress_Reduction\static\my_model.h5')

        # Make the prediction
        input_data = np.array([[respiration_rate, limb_movement, blood_oxygen, sleeping_hours, heart_rate]])
        stress_level = np.argmax(model.predict(input_data))

        # Render the result to the webpage
        return render(request, 'result.html', {'stress_level': stress_level, 'assistant_reply': assistant_reply})

    return render(request, 'index.html')
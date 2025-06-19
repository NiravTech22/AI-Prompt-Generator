from flask import Flask, request, jsonify
from prompt_generator import AIPromptGenerator

app = Flask(__name__)
generator = AIPromptGenerator()

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    words = data.get('words', [])
    try:
        result = generator.generate_prompt(words)
        return jsonify({"success": True, "prompt": result['prompt'], "history": result['history']})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 
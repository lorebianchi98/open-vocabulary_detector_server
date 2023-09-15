from flask import Flask, request

app = Flask(__name__)

def perform_hello_world(name):
    return "hello %s!" % name 


@app.route('/detect', methods=['POST'])
def detect_objects():
    if request.method == 'POST':
        data = request.data.decode('utf-8') 

        output = perform_hello_world(data)
        
        return output

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

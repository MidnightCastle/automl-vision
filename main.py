from tomato_disease_detection import app

if __name__ == '__main__':
    #app.run(debug=True)
    #print('hi')
    app.run(host='0.0.0.0', port=8080)
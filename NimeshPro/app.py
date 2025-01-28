from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__, template_folder='templates')
app.secret_key = 'your_secret_key'  # Required for flashing messages

# Route for the login page
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == 'admin' and password == '123':
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')

# Route for the stock market dashboard
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)

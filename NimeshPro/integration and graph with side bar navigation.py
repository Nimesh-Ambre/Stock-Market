from flask import Flask, render_template, request, redirect, url_for, flash, session
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(_name_, template_folder='templates')
app.secret_key = 'your_secret_key'  # Required for flashing messages

db_path = "C:/Users/nimesh/Downloads/naya/static/db.sqlite3"

def generate_stock_graph():
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT date, closing_price FROM stock_data"
        df = pd.read_sql_query(query, conn)
        conn.close()

        plt.figure(figsize=(10, 5))
        plt.plot(df['date'], df['closing_price'], label='Stock Price', color='blue')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.title('Stock Performance')
        plt.legend()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return plot_url
    except Exception as e:
        return None

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == 'admin' and password == '123':
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    stock_graph = generate_stock_graph()
    return render_template('dashboard.html', stock_graph=stock_graph)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

if _name_ == '_main_':
    app.run(debug=True)
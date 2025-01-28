@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    symbol = request.args.get('symbol', 'TATAMOTORS.NS')  # Default symbol
    stock_graph = generate_stock_graph(symbol)
    return render_template('dashboard.html', stock_graph=stock_graph)

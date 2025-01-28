def generate_stock_graph(symbol):
    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT date, closing_price FROM stock_data WHERE symbol='{symbol}'"
        df = pd.read_sql_query(query, conn)
        conn.close()

        plt.figure(figsize=(10, 5))
        plt.plot(df['date'], df['closing_price'], label=f'{symbol} Closing Prices', color='blue')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.title(f'{symbol} Chart')
        plt.legend()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return plot_url
    except Exception as e:
        return None

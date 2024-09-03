import yfinance as yf

def trigger_YH():
   
    company = ["MSFT", "TSLA", "AMZN", "META", "AAPL", "GOOGL", "NVDA"]
    for i in company:
        # print("\n \n \n hist")
        # get historical market data
        # hist = msft.history(period="1mo")
        # print("Historical Data: \n ", hist)
        ticker = yf.Ticker(i)

        # print("\n \n \n hist", i)
        # get historical market data
        # 1D 5D 3M 6M 1Y
        hist_6mo = ticker.history(period="6mo")
        hist_1mo = ticker.history(period="1mo")
        hist_5d = ticker.history(period="5d")
        hist_1d = ticker.history(period="1d")
        # print(f"Historical Data: \n 1 day stock:{hist_1d} /n /n 1 month stock:{hist_1mo} /n /n 5 day stock:{hist_5d}")
        # print("data type:", type(hist_1mo))
        hist_6mo.to_csv(f"data/{i}_hist_6mo.csv", index = False)
        hist_1mo.to_csv(f"data/{i}_hist_1mo.csv", index = False)
        hist_5d.to_csv(f"data/{i}_hist_5d.csv", index = False)
        hist_1d.to_csv(f"data/{i}_hist_1d.csv", index = False)
# trigger_YH()
# # show meta information about the history (requires history() to be called first)
# msft.history_metadata

# # show actions (dividends, splits, capital gains)
# msft.actions
# msft.dividends
# msft.splits
# msft.capital_gains  # only for mutual funds & etfs

# # show share count
# msft.get_shares_full(start="2022-01-01", end=None)

# # show financials:
# # - income statement
# msft.income_stmt
# msft.quarterly_income_stmt
# # - balance sheet
# msft.balance_sheet
# msft.quarterly_balance_sheet
# # - cash flow statement
# msft.cashflow
# msft.quarterly_cashflow
# # see `Ticker.get_income_stmt()` for more options

# # show holders
# msft.major_holders
# msft.institutional_holders
# msft.mutualfund_holders
# msft.insider_transactions
# msft.insider_purchases
# msft.insider_roster_holders

# # show recommendations
# msft.recommendations
# msft.recommendations_summary
# msft.upgrades_downgrades

# # Show future and historic earnings dates, returns at most next 4 quarters and last 8 quarters by default.
# # Note: If more are needed use msft.get_earnings_dates(limit=XX) with increased limit argument.
# msft.earnings_dates

# # show ISIN code - *experimental*
# # ISIN = International Securities Identification Number
# msft.isin

# # show options expirations
# msft.options

# # show news
# msft.news

# # get option chain for specific expiration
# opt = msft.option_chain('YYYY-MM-DD')
# # data available via: opt.calls, opt.puts
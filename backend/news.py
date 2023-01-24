from pygooglenews import GoogleNews

gn = GoogleNews()

# search for the best matching articles that mention MSFT and 
# do not mention AAPL (over the past 6 month
search = gn.search('SPY', when = '1m')

print(search['feed'].title)

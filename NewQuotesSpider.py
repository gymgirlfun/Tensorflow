import scrapy
import re

class NewQuotesSpider(scrapy.Spider):
	'''Extract all pages, and author profiles from another page'''
	name = "NewQuotesSpider"
	start_urls = ["http://quotes.toscrape.com/"]
	text = []
	author = []
	born_location = {}
	count_finish = 0
	author_index = 0  # it's not guaranteed that the yielded requests will execute in order

	def extract_born_city(self, response, index):
		city = response.xpath("//span[@class='author-born-location']/text()")
		if city:
			city = city[0].get()
			print("nshi test: ", city)
			city = re.sub(r"^in ", "", city)
			self.born_location[index] = city
			self.count_finish -= 1
		if self.count_finish == 0:
			self.summary()
			

	def parse(self, response):
		quotes = response.xpath("//div[@class='quote']")
		self.count_finish += len(quotes)
		for quote in quotes:
			self.text.append(quote.xpath("span[@class='text']/text()").get())
			self.author.append(quote.xpath("span/small/text()").get())
			about_author = quote.xpath("span/a/@href")
			if about_author:
				url = response.urljoin(about_author[0].extract())
				yield scrapy.Request(url, callback=self.extract_born_city, cb_kwargs=dict(index=self.author_index), dont_filter=True)
				self.author_index += 1

			
		next = response.xpath("//li[@class='next']/a/@href")
		if next:
			url = response.urljoin(next[0].extract())
			yield scrapy.Request(url, callback=self.parse)

	def summary(self):
		print("-----summary------")
		# print(len(self.text))
		# print(len(self.author))
		# print(len(self.born_location))
		size = len(self.text)
		if (len(self.author) != size or len(self.born_location) != size):
			raise Exception("The size of text, author, born_location not match.")
		item = []
		for index in range(0, size):
			item.append({"text": self.text[index], "author": self.author[index], "born_location": self.born_location[index]})
		print(item)


class Car:
    def __init__(self, name, money):
        self.name = name
        self.money = money

    def put(self):
        print(self.name, self.money)


class minicar(Car):
    def __init__(self, name, money, color):
        super().__init__(name, money)
        self.color = color

    def miniput(self):
        print(self.name, self.money, self.color)

    def put(self):
        print(self.name, self.money, self.color)


car = Car('haha', 500)
minicar0 = minicar('hehe', 600, 'red')
car.put()
minicar0.put()
minicar0.miniput()

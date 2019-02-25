import json
import time

class Fack_turtle():
    def __init__(self):
        self.para = {'time': time.time()}


    def move_now(self, vec, rad):
        while True:
            file = open("index_cut.txt", "r")
            while True:
                try:
                    para = json.loads(file.read())
                    break
                except:
                    continue
            file.close()
            if vec != 0.0 or rad != 0.0:
                print('moving!!', vec, rad)
            if para['time'] != self.para['time']:
                self.para = para
                print(self.para)
                return
            time.sleep(0.1)

    def go_frover(self):
        while True:
            if 'camera' in self.para:
                vec = (50000.0 - self.para['camera']['length'])/ 30000.0
                rad = (360.0 - self.para['camera']['location']) / 360.0 + 0.45
            else:
                vec = 0.0
                rad = 0.0
            self.move_now(vec, rad)


if __name__ == "__main__":
    fly_turtle = Fack_turtle()
    fly_turtle.go_frover()
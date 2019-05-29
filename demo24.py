import copy
a = ['A','K']
b = ['Q','J']
user1 = [a,b]
user2 = user1
user3 = copy.copy(user1)
user4 = copy.deepcopy(user1)
#ctrl+alt+M => 程式抽成函數
def displayList(message) :
    print(message)
    print('Uaer1:',user1)
    print('Uaer2:',user2)
    print('Uaer3:',user3)
    print('Uaer4:',user4)
    print('User1:',hex(id(user1)),'User2:',hex(id(user2)))
    print('User3:',hex(id(user3)),'User4:',hex(id(user4)))
displayList("First")

a[0] = 10
displayList("Second")

user1.append("joker")
displayList("Third")


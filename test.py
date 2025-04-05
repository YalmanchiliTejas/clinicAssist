
def main(orderlist):

    prime = []
    non_prime = []

    for i in range(len(orderlist)):
        temp = orderlist[i].split(" ")
        if not temp[1][0].isdigit():
            prime.append(([temp[0], temp[1:]]))
        else:
            non_prime.append(orderlist[i])
    

    prime = sorted(prime, key=lambda x: (x[1], x[0]))
    

    ans = []

    for i in prime:
        ans.append(i[0] + " " + " ".join(i[1]))
    for i in non_prime:
        ans.append(i)
    return ans


if __name__ == "__main__":
    orderList = ["zld 93 12", "fp kindle book", "10a echo show", "17g 12 25 6",  "ab1 kindle book", "125 echo dot second generation"]
    print(main(orderList))

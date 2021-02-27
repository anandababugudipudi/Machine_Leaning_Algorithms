# =============================================================================== #
#                             Guess The Word Game                                 #
# =============================================================================== #
# When the user plays WordGuess, the computer first selects a secret word at      #
# random from a list which was obtained from a txt file. The program then prints  #
# out a row of dashes—one for each letter in the secret word and asks the user to #
# guess a letter. If the user guesses a letter that is in the word, the word is   #
# redisplayed with all instances of that letter shown in the correct positions,   #
# along with any letters correctly guessed on previous turns. If the letter does  #
# not appear in the word, the user is charged with an incorrect guess. The user   #
# keeps guessing letters until either:                                            #
#       (1) the user has correctly guessed all the letters in the word or         #
#       (2) the user has made ten incorrect guesses.                              #
#                                                                                 #
# Author: anandababugudipudi@gmail.com                                            #
#                                                                                 #
# =============================================================================== #

import random
def strToCharList(word):
    return [char for char in word]

fileRead = open("words.txt", "r")
words = fileRead.readlines()
hiddenWord = words[random.randint(0, len(words)-1)].rstrip().upper()
hiddenWordList = strToCharList(hiddenWord)
attempts = 10
suggestedWord = "-"*len(hiddenWord)
suggestedWordList = strToCharList(suggestedWord)
print(f"The word now looks like this: {suggestedWord}")
print(f"You have {attempts} guesses left.")

while (attempts > 0):
    attempts -= 1
    ch = input("Type a single letter here, then press Enter: ")
    ch = ch.upper()
    if (ch in  hiddenWordList):
        for i in range(len(suggestedWordList)):
           if (hiddenWordList[i] == ch):
               suggestedWordList[i] = ch
        print("Your guess is correct.")
    else:
        print(f"There are no {ch}'s in the word.")
    if (hiddenWord.upper() == "".join(suggestedWordList).upper()):
        print(f"\nCongratulations, the word is: {hiddenWord}")
        break
    print(f"The word now looks like this:  {''.join(suggestedWordList)}")
    
    print(f"You have {attempts} guesses left.")
if (attempts == 0):
    print(f"Sorry, you lost. Better Luck next time. \nThe secret word was: {hiddenWord}")        
        
    



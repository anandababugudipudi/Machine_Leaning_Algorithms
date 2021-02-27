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
# Defining necessary functions
# Function to split a string into characters list
def strToCharList(word):
    return [char for char in word]

# Reading a random word from the file
fileRead = open("words.txt", "r")
words = fileRead.readlines()
# Generating the hidden word randomaly from words
hiddenWord = words[random.randint(0, len(words)-1)].rstrip().upper()
hiddenWordList = strToCharList(hiddenWord)
# Declaring necessary variables
attempts = 10
suggestedWord = "-"*len(hiddenWord)
suggestedWordList = strToCharList(suggestedWord)

print(f"The word now looks like this: {suggestedWord}")
print(f"You have {attempts} guesses left.")

# Taking character inputs from the user
while (attempts > 0):
    # Reducing the attempts by 1 for every iteration
    attempts -= 1
    # Taking characters from user and converting it into upper
    ch = input("Type a single letter here, then press Enter: ")
    ch = ch.upper()
    # Chicking whether the character is in the hidden word or not
    # If it is in the hidden word then changing the '-' in suggested list to this character
    if (ch in  hiddenWordList):
        for i in range(len(suggestedWordList)):
           if (hiddenWordList[i] == ch):
               suggestedWordList[i] = ch
        # Printing the output if the guress is correct   
        print("Your guess is correct.")
    else:
        # Printing the output if the guress is wrong
        print(f"There are no {ch}'s in the word.")
        
    # If the word is correctly guessed we will show Congratulations message
    if (hiddenWord.upper() == "".join(suggestedWordList).upper()):
        print(f"\nCongratulations, the word is: {hiddenWord}")
        break
    # Showing the hints and attempts left
    print(f"The word now looks like this:  {''.join(suggestedWordList)}")
    print(f"You have {attempts} guesses left.")
# Showing the user what the hidden word was
if (attempts == 0):
    print(f"Sorry, you lost. Better Luck next time. \nThe secret word was: {hiddenWord}")        
        
    



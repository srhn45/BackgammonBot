## MCTS Based Backgammon Bot & Trainer

### Includes:
1. A pretrained shallow resnet based bot.
2. An MCTS based training algorithm.
3. Self built implementation of backgammon in a lightweight format.
4. PyGame implementation of the game where you can play against a valid PyTorch model.

*Training still ongoing.*

*Currently the model almost always converges into short-term loss probability minimization by only moving the pieces that are closest to end columns, which I'm guessing is because they're less likely to break. As I'm training on my own system, I can't easily fix this by increasing simulation depth or using a deeper model. **Any suggestions are welcome!***

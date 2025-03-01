"""
A collection of chess opening positions in FEN format for use with the chess engine.
"""

# Dictionary of opening positions with names and FENs
OPENING_POSITIONS = {
    # Open Games
    "Starting Position": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "Ruy Lopez": "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "Italian Game": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "Sicilian Defense": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "French Defense": "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
    "Caro-Kann": "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
    
    # Semi-Open Games
    "Sicilian Najdorf": "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    "Sicilian Dragon": "r1bqkb1r/pp2pp1p/2np1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 7",
    "French Winawer": "rnbqk1nr/ppp2ppp/4p3/3p4/1b1PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 2 4",
    
    # Closed Games
    "Queen's Gambit": "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
    "Queen's Gambit Accepted": "rnbqkbnr/ppp1pppp/8/8/2pP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
    "Queen's Gambit Declined": "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
    "Slav Defense": "rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
    "King's Indian Defense": "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
    "Nimzo-Indian Defense": "rnbqk2r/pppp1ppp/4pn2/8/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq - 2 4",
    
    # Advanced Positions
    "Ruy Lopez Berlin": "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "Sicilian Najdorf English Attack": "r2q1rk1/1p1nbppp/p3b3/4p3/4P3/1NN2P2/PPPQ2PP/2KR1B1R w - - 5 11",
    "QGD Tartakower": "r2q1rk1/pbpnbpp1/1p3n1p/3p4/3P4/1B1BPN2/PP3PPP/RN1Q1RK1 w - - 4 11",
    "KID Classical (Petrosian)": "r1bq1rk1/pppnnppp/3p4/3Pp3/2P1P3/5N2/PP2BPPP/RNBQ1RN1 w - - 3 10",
    "Caro-Kann Classical": "r3kbnr/ppq1ppp1/2p4p/7P/3P4/3Q1N2/PPP2PP1/R1B1K2R w KQkq - 1 11",
    
    # Famous Positions
    "Immortal Game": "r1bk3r/p2pBpNp/n4n2/1p1NP2P/6P1/3P4/P1P1K3/q5b1 b - - 1 21",
    "Opera Game": "r1bk3r/pp1pBp1p/2p2p2/4P3/8/P7/1PP2PPP/2KR4 b - - 0 18"
}

def get_opening_names():
    """Returns a list of all opening names."""
    return list(OPENING_POSITIONS.keys())

def get_opening_fen(name):
    """Returns the FEN for a specific opening by name."""
    return OPENING_POSITIONS.get(name)

def get_random_opening():
    """Returns a random opening (name, fen) tuple."""
    import random
    name = random.choice(list(OPENING_POSITIONS.keys()))
    return (name, OPENING_POSITIONS[name])
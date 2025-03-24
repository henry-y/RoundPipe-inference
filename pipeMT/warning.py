warning_thrown = set()

def throw(info: str):
    if info in warning_thrown:
        return
    warning_thrown.add(info)
    print('[pipeMT WARNING]', info)

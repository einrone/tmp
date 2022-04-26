from pathlib import Path
from typing import Union, List, Optional, Dict


class FetchPath:
    def __init__(self):
        pass

    def __call__(
        self, filepath: Union[str, Path], filterkey: Optional[str] = None
    ) -> List[Path]:
        """
        Class which is responsible to fetch desired files for
        a given path.

        args:
            filepath: Location of the desired file
            filterkey: the key to filter out undesired files
        return:
            returns the desired path of the file one wish
            to have

        """
        filepath = Path(filepath)

        if filterkey is None:
            filterkey = "*"
        else:
            pass

        paths = list(filepath.glob(f"{filterkey}"))

        if len(paths) == 1:
            return paths[0]
        else:
            return paths

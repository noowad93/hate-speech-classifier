import pytest
from abusing.utils import load_data

TEST_FILE_INPUT = """comments\tcontain_gender_bias\tbias\thate
안녕하세요 저는 정다운입니다.\tFalse\tnone\tnone\n"""

@pytest.mark.parametrize(
    "input_data, index,expected_result",
    [
        pytest.param(
            TEST_FILE_INPUT,
            0,
            ("안녕하세요 저는 정다운입니다.",0,0)
        ),
    ],
)
def test_load_data(tmpdir, input_data, index, expected_result):
    test_path = tmpdir.join("test_data.tsv")
    test_path.write(input_data)

    result = load_data(test_path)
    assert result[index] == expected_result

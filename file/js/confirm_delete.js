confirm_and_submit = function () {undefined
    let delete_form = document.getElementById('delete_form');
    let real_name = document.getElementsByName('name');
    let input_name = document.getElementsByName('inputname');
    let msg = document.getElementsByName('msg')
    console.log(456);
    if (real_name.value === input_name.value) {
        delete_form.submit();
    } else {
        msg.innerHTML = '输入错误！'
    }
}

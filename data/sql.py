import pymysql


def Select():
    connection = pymysql.connect("localhost", "root", "root", "ocr", charset='utf8')

    cursor = connection.cursor()
    cursor.execute("select * from invoice")

    result = cursor.fetchall()

    connection.close()

    return result

def Select_h():
    connection = pymysql.connect("localhost", "root", "root", "ocr", charset='utf8')

    cursor = connection.cursor()
    cursor.execute("select * from h_invoice")

    result = cursor.fetchall()

    connection.close()

    return result


def Select_id(ids):
    connection = pymysql.connect("localhost", "root", "root", "ocr", charset='utf8')

    cursor = connection.cursor()
    sql = "select * from invoice where id = %d" %(ids)
    # print(sql)
    cursor.execute(sql)

    result = cursor.fetchall()

    connection.close()

    return result

def delect_id(ids):
    conn = pymysql.connect("localhost", "root", "root", "ocr", charset='utf8')
    cs1 = conn.cursor()
    # 执行sql语句
    query = "DELETE FROM invoice where id = %d" %(ids)

    print(query)
    try:
        cs1.execute(query)
        conn.commit()
    except:
        conn.rollback()

    cs1.close()
    conn.close()

def delect_id_h(ids):
    conn = pymysql.connect("localhost", "root", "root", "ocr", charset='utf8')
    cs1 = conn.cursor()
    # 执行sql语句
    query = "DELETE FROM h_invoice where id = %d" %(ids)

    print(query)
    try:
        cs1.execute(query)
        conn.commit()
    except:
        conn.rollback()

    cs1.close()
    conn.close()

def counts():
    connection = pymysql.connect("localhost", "root", "root", "ocr", charset='utf8')

    cursor = connection.cursor()
    sql = "SELECT COUNT('id')  FROM invoice"
    # print(sql)
    cursor.execute(sql)

    result = cursor.fetchall()

    connection.close()

    return result


def updata(invoiceNum_area,keyNum_area,name_X_block,ID_X_block,address_telephone_X_block,bankInfo_X_block,name_G_block,ID_G_block,address_telephone_G_block,bankInfo_G_block,commodityInfo_area,unit_area,
           numbers_area,retailPriceTax_area,amountOfMoneyTax_area,taxRate_area,taxAmount_area,total_area,specificationType_area,
           validationCode_keys,date_area,remark_keys,total_amount,total_tax,times,yz_state,id):
    conn = pymysql.connect("localhost", "root", "root", "ocr", charset='utf8')
    cs1 = conn.cursor()
    # 执行sql语句
    query = "UPDATE invoice SET invoiceNum_area="+ "'" + invoiceNum_area +"'"+",keyNum_area="+ "'" + keyNum_area +"'"+\
            ",name_X_block="+ "'" + name_X_block +"'"+",ID_X_block="+ "'" + ID_X_block +"'"\
            ",address_telephone_X_block="+ "'" + address_telephone_X_block +"'"+",bankInfo_X_block="+ "'" + bankInfo_X_block +"'"+\
            ",name_G_block="+ "'" + name_G_block +"'"+",ID_G_block="+ "'" + ID_G_block +"'"+\
            ",address_telephone_G_block="+ "'" + address_telephone_G_block +"'"+",bankInfo_G_block="+ "'" + bankInfo_G_block +"'"+\
            ",commodityInfo_area="+ "'" + commodityInfo_area +"'"+",unit_area="+ "'" + unit_area +"'"+\
            ",numbers_area ="+ "'" + numbers_area +"'"+",retailPriceTax_area="+ "'" + retailPriceTax_area +"'"+\
            ",amountOfMoneyTax_area="+ "'" + amountOfMoneyTax_area +"'"+",taxRate_area="+ "'" + taxRate_area +"'"+\
            ",taxAmount_area="+ "'" + taxAmount_area +"'"+",total_area="+ "'" + total_area +"'"+\
            ",specificationType_area="+ "'" + specificationType_area +"'"+",validationCode_keys ="+ "'" + validationCode_keys +"'"+\
            ",date_area="+ "'" + date_area +"'"+ ",total_amount="+ "'" + total_amount +"'"+ \
            ",remark_keys="+ "'" + remark_keys +"'"+ ",total_tax ="+"'"+ total_tax +"'"  \
            ",times="+ "'" + times + "'" + ",yz_state="+  str(yz_state)  + " "+"WHERE id = "+ "'" + str(id) + "'";


    print()
    print(query)
    try:
        cs1.execute(query)
        conn.commit()
    except:
        conn.rollback()

    cs1.close()
    conn.close()


def yz_states(yz_state,id):
    conn = pymysql.connect("localhost", "root", "root", "ocr", charset='utf8')
    cs1 = conn.cursor()
    # 执行sql语句
    query = "UPDATE invoice SET yz_state="+  str(yz_state)  + " "+"WHERE id = "+ "'" + str(id) + "'";


    print()
    print(query)
    try:
        cs1.execute(query)
        conn.commit()
    except:
        conn.rollback()

    cs1.close()
    conn.close()

# yz_states('545','455')

def Select_Numkey(key):
    connection = pymysql.connect("localhost", "root", "root", "ocr", charset='utf8')

    cursor = connection.cursor()
    sql = "select * from invoice where invoiceNum_area = %s" %(key)
    # print(sql)
    cursor.execute(sql)

    result = cursor.fetchall()

    connection.close()

    return result



def updata_stait(ti):
    conn = pymysql.connect("localhost", "root", "root", "ocr", charset='utf8')
    cs1 = conn.cursor()
    # 执行sql语句
    query = "UPDATE invoice SET yz_state="+  str(1) + ',times='+ "'" + ti + "'" + " "+"WHERE yz_state = "+ str(-1)

    print(query)
    try:
        cs1.execute(query)
        conn.commit()
    except:
        conn.rollback()

    cs1.close()
    conn.close()


def updata_stait2(ti):
    conn = pymysql.connect("localhost", "root", "root", "ocr", charset='utf8')
    cs1 = conn.cursor()
    # 执行sql语句
    query = "UPDATE invoice SET yz_state="+  str(-2) + ',times='+ "'" + ti + "'" + " "+"WHERE yz_state = "+ str(0)

    print(query)
    try:
        cs1.execute(query)
        conn.commit()
    except:
        conn.rollback()

    cs1.close()
    conn.close()

def delect_state():
    conn = pymysql.connect("localhost", "root", "root", "ocr", charset='utf8')
    cs1 = conn.cursor()
    # 执行sql语句
    query = "DELETE FROM invoice where state = 1";

    print(query)
    try:
        cs1.execute(query)
        conn.commit()
    except:
        conn.rollback()

    cs1.close()
    conn.close()

def inter_h_invocie(set_out,arrive,train_number,data,time,price,name,flg='0'):
    conn = pymysql.connect("localhost", "root", "root", "ocr", charset='utf8')
    cs1 = conn.cursor()
    # 执行sql语句
    query = "INSERT INTO h_invoice(set_out, arrive, train_number, datas, times, price, namess,flg ) VALUES (" + "'" + set_out + "'" + "," + "'" + arrive + "'" + "," + "'" + train_number + "'" + "," + "'" + data + "'" + "," + \
                  "'" + time + "'" + "," + "'" + price + "'" + "," + "'" + name + "'" + "," + "'" + flg + "'" +")"

    print()
    print(query)
    try:
        cs1.execute(query)
        conn.commit()
    except:
        conn.rollback()

    cs1.close()
    conn.close()

def Select_stati0():
    connection = pymysql.connect("localhost", "root", "root", "ocr", charset='utf8')

    cursor = connection.cursor()
    cursor.execute("SELECT * FROM invoice WHERE yz_state = 0")

    result = cursor.fetchall()
    print(len(result))
    connection.close()

    return result

def Select_stati1():
    connection = pymysql.connect("localhost", "root", "root", "ocr", charset='utf8')

    cursor = connection.cursor()
    cursor.execute("SELECT * FROM invoice WHERE yz_state = 1")

    result = cursor.fetchall()
    # print(result)
    connection.close()

    return result

def Select_stati_2():
    connection = pymysql.connect("localhost", "root", "root", "ocr", charset='utf8')

    cursor = connection.cursor()
    cursor.execute("SELECT * FROM invoice WHERE yz_state = -2")

    result = cursor.fetchall()
    # print(result)
    connection.close()

    return result

def Select_statih0():
    connection = pymysql.connect("localhost", "root", "root", "ocr", charset='utf8')

    cursor = connection.cursor()
    cursor.execute("SELECT * FROM h_invoice WHERE flg = 0")

    result = cursor.fetchall()
    # print(result)
    connection.close()

    return result

def Select_statih1():
    connection = pymysql.connect("localhost", "root", "root", "ocr", charset='utf8')

    cursor = connection.cursor()
    cursor.execute("SELECT * FROM h_invoice WHERE flg = 1")

    result = cursor.fetchall()
    # print(result)
    connection.close()

    return result


def select_user(name,pwd):
    connection = pymysql.connect("localhost", "root", "root", "ocr", charset='utf8')

    cursor = connection.cursor()

    sql = "SELECT * FROM USER WHERE NAME = "+ "'" + name + "'"  + "AND" + "'"+ pwd +  "'";
    print(sql)
    cursor.execute(sql)

    result = cursor.fetchall()

    print(result)

    connection.close()

    return result

def insrt_into_invoice(strData,whole_path):
    print(strData,whole_path)
    print('=================================================  S Q L   ==============================================================')
    data1 = '=='.join(strData['发票类型'])
    data2 = '=='.join(strData['发票号码'])
    data3 = '=='.join(strData['发票代码'])
    data4 = '=='.join(strData['销售方名称'])
    data5 = '=='.join(strData['销售方识别号'])
    data6 = '=='.join(strData['销售方地址和电话'])
    data7 = '=='.join(strData['销售方银行信息'])
    data8 = '=='.join(strData['购货方名称'])
    data9 = '=='.join(strData['购货方识别号'])
    data10 = '=='.join(strData['购货方地址和电话'])
    data11 = '=='.join(strData['购货方银行信息'])
    data12 = '=='.join(strData['商品信息(品名/规格/产地)'])
    data13 = '=='.join(strData['单位'])
    data14 = '=='.join(strData['数量'])
    data15 = '=='.join(strData['单价'])
    data16 = '=='.join(strData['金额'])
    data17 = '=='.join(strData['税率'])
    data18 = '=='.join(strData['税额'])
    data19 = '=='.join(strData['发票总额'])
    data20 = '=='.join(strData['规格型号'])
    data21 = '=='.join(strData['效验码'])
    data22 = '=='.join(strData['开票日期'])
    data23 = '=='.join(strData['收款人'])
    data24 = '=='.join(strData['复核人'])
    data25 = '=='.join(strData['开票人'])
    data26 = '=='.join(strData['备注'])
    data27 = '=='.join(strData['金额合计'])
    data28 = '=='.join(strData['税额合计'])

    print()
    # print(data16)

    import pymysql
    from data import sql
    num = sql.counts()
    num = int(num[0][0] + 1)
    imagePath = whole_path

    print()

    for dm in data2:
        if dm.isdigit() != True and dm != '=':
            if dm == '己' or dm == '已' and dm != '=':
                data2 = data2.replace(dm, '2')
            else:
                data2 = data2.replace(dm, '')

    for dm in data3:
        if dm.isdigit() != True and dm != '=':
            data3 = data3.replace(dm, '')

    for dm in data27:
        if dm.isdigit() != True and dm != '.' and dm != '-':
            if dm == ',' or dm == '。':
                data27 = data27.replace(dm, '.')
            else:
                data27 = data27.replace(dm, '')
    flag = False
    for dm in data28:
        if dm.isdigit() != True and dm != '.' and dm != '-':
            if dm == ',' or dm == '。' and flag == False:
                flag = True
                data28 = data28.replace(dm, '.')
            else:
                data28 = data28.replace(dm, '')

    for dm in data14:
        if dm.isdigit() != True and dm != '.' and dm != '=':
            data14 = data14.replace(dm, '')

    for dm in data15:
        if dm.isdigit() != True and dm != '.' and dm != '=':
            data15 = data15.replace(dm, '')

    for dm in data16:
        if dm.isdigit() != True and dm != '.' and dm != '=':
            data16 = data16.replace(dm, '')

    for dm in data17:
        if dm.isdigit() != True and dm != '%' and dm != '=':
            data17 = data17.replace(dm, '')

    for dm in data18:
        if dm.isdigit() != True and dm != '.' and dm != '=':
            data18 = data18.replace(dm, '')

    for dm in data19:
        if dm.isdigit() != True and dm != '.' and dm != '-':
            if dm == ',' or dm == '。':
                data19 = data19.replace(dm, '.')
            else:
                data19 = data19.replace(dm, '')

    for dm in data26:
        if dm.isdigit() != True and dm != '=':
            if dm == '日':
                data26 = data26.replafce(dm, '8')
            else:
                data26 = data26.replace(dm, '')

    for dm in data22:
        if dm.isdigit() != True and dm != '.' and dm != '==':
            if dm == 'g' or dm == 'v':
                data22 = data22.replace(dm, '8')
            else:
                data22 = data22.replace(dm, '')


    fg = sql.Select_Numkey(data2[-8:])
    db = pymysql.connect("localhost", "root", "root", "ocr", charset='utf8')
    cursor = db.cursor()
    if len(fg) == 0:
        sql = "INSERT INTO invoice(invoice_type,invoiceNum_area,keyNum_area,name_X_block,ID_X_block,address_telephone_X_block,bankInfo_X_block,name_G_block,ID_G_block,address_telephone_G_block,bankInfo_G_block,commodityInfo_area,unit_area,numbers_area,retailPriceTax_area,amountOfMoneyTax_area,taxRate_area,taxAmount_area,total_area,specificationType_area,validationCode_keys,date_area,payee_area,reviewer_area,drawer_area,remark_keys,total_amount,total_tax,img_name,times,yz_state,state,yyz_state) VALUES(" + "'" + data1 + "'" + "," + "'" + data2[
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              -8:] + "'" + "," + "'" + data3 + "'" + "," + "'" + data4 + "'" + "," + \
              "'" + data5 + "'" + "," + "'" + data6 + "'" + "," + "'" + data7 + "'" + "," + "'" + data8 + "'" + "," + "'" + data9 + "'" + "," + "'" + data10 + "'" + "," + \
              "'" + data11 + "'" + "," + "'" + data12 + "'" + "," + "'" + data13 + "'" + "," + "'" + data14 + "'" + "," + "'" + data15 + "'" + "," + "'" + data16 + "'" + "," + \
              "'" + data17 + "'" + "," + "'" + data18 + "'" + "," + "'" + str(
            data19) + "'" + "," + "'" + data20 + "'" + "," + "'" + data21 + "'" + "," + "'" + data22 + "'" + "," + \
              "'" + data23 + "'" + "," + "'" + data24 + "'" + "," + "'" + data25 + "'" + "," + "'" + data26 + "'" + "," + "'" + str(
            data27) + "'" + "," + "'" + str(data28) + "'" + "," + \
              "'" + imagePath + "'" + "," + "'" + '' + "'" + "," + str(0) + "," + str(0) + "," + str(0) + ")"
        print(sql)
    else:
        sql = "INSERT INTO invoice(invoice_type,invoiceNum_area,keyNum_area,name_X_block,ID_X_block,address_telephone_X_block,bankInfo_X_block,name_G_block,ID_G_block,address_telephone_G_block,bankInfo_G_block,commodityInfo_area,unit_area,numbers_area,retailPriceTax_area,amountOfMoneyTax_area,taxRate_area,taxAmount_area,total_area,specificationType_area,validationCode_keys,date_area,payee_area,reviewer_area,drawer_area,remark_keys,total_amount,total_tax,img_name,times,yz_state,state,yyz_state) VALUES(" + "'" + data1 + "'" + "," + "'" + data2[
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              -8:] + "'" + "," + "'" + data3 + "'" + "," + "'" + data4 + "'" + "," + \
              "'" + data5 + "'" + "," + "'" + data6 + "'" + "," + "'" + data7 + "'" + "," + "'" + data8 + "'" + "," + "'" + data9 + "'" + "," + "'" + data10 + "'" + "," + \
              "'" + data11 + "'" + "," + "'" + data12 + "'" + "," + "'" + data13 + "'" + "," + "'" + data14 + "'" + "," + "'" + data15 + "'" + "," + "'" + data16 + "'" + "," + \
              "'" + data17 + "'" + "," + "'" + data18 + "'" + "," + "'" + str(
            data19) + "'" + "," + "'" + data20 + "'" + "," + "'" + data21 + "'" + "," + "'" + data22 + "'" + "," + \
              "'" + data23 + "'" + "," + "'" + data24 + "'" + "," + "'" + data25 + "'" + "," + "'" + data26 + "'" + "," + "'" + str(
            data27) + "'" + "," + "'" + str(data28) + "'" + "," + \
              "'" + imagePath + "'" + "," + "'" + '' + "'" + "," + str(0) + "," + str(1) + "," + str(0) + ")"
        print(sql)

    try:
        cursor.execute(sql)
        db.commit()
    except:
        db.rollback()
    db.close()


# select_user('admin','123456')





# Select_stati0()
# inter_h_invocie('1','1','1','1','10','1','1')
# da = Select()
# print(da[0][0])




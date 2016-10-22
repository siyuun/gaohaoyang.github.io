---
layout: post
title:  "扫雷小游戏VB"
date:   2016-10-10 22:40:04
categories: VB
tags: game
---

* content
{:toc}

**想题目是什么想了半天 什么导弹炸弹 原来是扫雷啊**
老师vb写的，自己改了点

		Public Sub New(ByVal m As Integer, ByVal n As Integer)
			MineMatrix = New clsMineMatrix(m, n, 10)
			Buttons = New List(Of Button)
			Me.M = m : Me.N = n
			For i = 0 To m - 1
				For j = 0 To n - 1
					Dim ButtonX As Button = New Button
					With ButtonX
						.Location = New System.Drawing.Point(MarginX + w * j, MarginY + i * w)
						.Name = "Button " & i & " " & j  'Button 2 3
						.Size = New System.Drawing.Size(w, w)
						.Text = ""
						.UseVisualStyleBackColor = True
						AddHandler ButtonX.Click, AddressOf ButtonX_Click
					End With
					Buttons.Add(ButtonX)
				Next
			Next
		End Sub
		 Private Sub ButtonX_Click(ByVal sender As Object, ByVal e As EventArgs)
			Dim ij() As String = sender.name.split(" ")
			Dim i As Integer = ij(1), j As Integer = ij(2)
			MsgBox(i & "," & j)
		End Sub

要添加的功能：
+ 点击显示 周围八个数字或是雷
+ 点击 到雷 游戏结束 显示全部
+ 把位置在右边的编辑框中显示出来